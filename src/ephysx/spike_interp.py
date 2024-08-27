import dataclasses
import threading
import time
from typing import Optional, Union

from tqdm.auto import tqdm, trange

import h5py
import joblib
import numba
import numpy as np
import torch
import torch.nn.functional as F
from dartsort.cluster import density, initial
from dartsort.cluster.modes import smoothed_dipscore_at
from dartsort.config import ClusteringConfig
from dartsort.util import data_util, drift_util, spiketorch, waveform_util
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.interpolate import PchipInterpolator
from scipy.sparse import coo_array, dok_array
from scipy.spatial import KDTree
from sklearn.gaussian_process.kernels import RBF

tqdm_kw = dict(smoothing=0, mininterval=1 / 24)


class timer:
    def __init__(self, name="timer", format="{:0.1f}"):
        self.name = name
        self.format = format

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.t = time.perf_counter() - self.start
        t = self.format.format(self.t)
        print(self.name, "took", t, "s")


# -- helper classes


class SpikeData(torch.nn.Module):
    """Data bag to keep things tidy."""

    def __init__(
        self,
        n_chans_full: int,
        waveform_rank: int,
        n_chans_unit: int,
        n_chans_waveform: int,
        n_chans_reassign: int,
        n_spikes: int,
        keepers: np.ndarray,
        spike_train: data_util.DARTsortSorting,
        motion_est,
        channels: np.ndarray,
        times_samples: np.ndarray,
        times_seconds: np.ndarray,
        original_channel_index: np.ndarray,
        reassign_channel_index: np.ndarray,
        registered_original_channel_index: torch.tensor,
        registered_reassign_channel_index: torch.tensor,
        static_main_channels: torch.tensor,
        cluster_channel_index: np.ndarray,
        amp_vecs: torch.Tensor,
        amps: np.ndarray,
        original_tpca_embeds: Union[np.ndarray, h5py.Dataset],
        reassign_tpca_embeds: Optional[np.ndarray],
        original_static_channels: np.ndarray,
        reassign_static_channels: np.ndarray,
        registered_geom: np.ndarray,
        geom: np.ndarray,
        in_memory: bool,
        pin: bool = True,
        on_device: bool = False,
        tpca: Optional["TemporalPCAFeaturizer"] = None,
        kriging_sigma: Optional[float] = None,
    ):
        super().__init__()
        self.n_chans_full = n_chans_full
        self.waveform_rank = waveform_rank
        self.n_chans_unit = n_chans_unit
        self.n_chans_waveform = n_chans_waveform
        self.n_chans_reassign = n_chans_reassign
        self.n_spikes = n_spikes
        self.in_memory = in_memory
        self.tpca = tpca
        self.geom = geom
        self.times_samples = times_samples
        self.pin = pin
        self.kriging_sigma = kriging_sigma

        self.spike_train = spike_train
        self.motion_est = motion_est

        # arrays not needed in torch
        self.original_channel_index = original_channel_index
        self.reassign_channel_index = reassign_channel_index
        self.static_main_channels = torch.as_tensor(static_main_channels)

        # CPU tensors
        self.keepers = torch.from_numpy(keepers)
        self.amps = amps
        original_static_channels = torch.as_tensor(original_static_channels)
        reassign_static_channels = torch.as_tensor(reassign_static_channels)
        if in_memory:
            original_tpca_embeds = torch.as_tensor(original_tpca_embeds)
            reassign_tpca_embeds = torch.as_tensor(reassign_tpca_embeds)
            amp_vecs = torch.as_tensor(amp_vecs)

        # these need to be on the GPU for speed
        # this could easily become an issue. in that case, they will still be small
        # enough to fit into ram, and the solution is to implement some kind of chunking
        # in reassign(). i'm talking huge chunks -- looping over .batch_size is slow.
        self.register_buffer("reassign_static_channels", reassign_static_channels)
        self.register_buffer("reassign_tpca_embeds", reassign_tpca_embeds)

        if on_device and self.in_memory:
            self.register_buffer("original_tpca_embeds", original_tpca_embeds)
            self.register_buffer("original_static_channels", original_static_channels)
            self.register_buffer("amp_vecs", amp_vecs)
        elif pin and self.in_memory and torch.cuda.is_available():
            self.original_tpca_embeds = original_tpca_embeds.pin_memory()
            # self.reassign_tpca_embeds = reassign_tpca_embeds.pin_memory()
            self.original_static_channels = original_static_channels.pin_memory()
            self.amp_vecs = amp_vecs.pin_memory()
        else:
            self.original_tpca_embeds = original_tpca_embeds
            # self.reassign_tpca_embeds = reassign_tpca_embeds
            self.original_static_channels = original_static_channels
            self.amp_vecs = amp_vecs

        # GPU
        self.register_buffer("registered_geom", torch.as_tensor(registered_geom, dtype=self.original_tpca_embeds.dtype))
        self.register_buffer("padded_registered_geom", F.pad(self.registered_geom, (0, 0, 0, 1), value=torch.nan))
        self.register_buffer(
            "cluster_channel_index", torch.as_tensor(cluster_channel_index)
        )
        self.register_buffer(
            "times_seconds", torch.as_tensor(times_seconds, dtype=torch.float)
        )
        self.register_buffer("channels", torch.as_tensor(channels))
        self.register_buffer(
            "registered_original_channel_index", registered_original_channel_index
        )
        self.register_buffer(
            "registered_reassign_channel_index", registered_reassign_channel_index
        )

    def get_waveforms(self, index, device=None, kind="original"):
        if self.in_memory:
            if kind == "original":
                waveforms = self.original_tpca_embeds
            elif kind == "reassign":
                waveforms = self.reassign_tpca_embeds
            transfer = torch.is_tensor(waveforms)
            transfer = (
                transfer and torch.is_tensor(index) and index.device != waveforms.device
            )
            if transfer:
                index = index.cpu()
            waveforms = waveforms[index]
        elif kind == "reassign":
            index = index.cpu()
            scalar = np.isscalar(index)
            index = np.atleast_1d(index)
            indices = self.keepers[index]
            mask = np.zeros(self.original_tpca_embeds.shape[0], dtype=bool)
            mask[indices] = 1

            waveforms = _channel_subset_by_chunk(
                mask,
                self.original_tpca_embeds,
                self.channels[index],
                self.original_channel_index,
                self.reassign_channel_index,
            )
            if scalar:
                waveforms = waveforms[0]
            waveforms = torch.from_numpy(waveforms)
        elif kind == "original":
            index = index.cpu()
            scalar = np.isscalar(index)
            index = np.atleast_1d(index)
            indices = self.keepers[index]
            mask = np.zeros(self.original_tpca_embeds.shape[0], dtype=bool)
            mask[indices] = 1

            waveforms = _read_by_chunk(
                mask,
                self.original_tpca_embeds,
                show_progress=False,
            )
            if scalar:
                waveforms = waveforms[0]
            waveforms = torch.from_numpy(waveforms)

        if device is not None:
            waveforms = waveforms.to(device, non_blocking=self.pin)
        return waveforms

    def get_static_amp_vecs(
        self,
        index,
        device=None,
    ):
        amp_vecs = self.amp_vecs[index]
        reassign_static_channels = self.reassign_static_channels[index]
        amp_vecs = drift_util.grab_static(
            amp_vecs,
            reassign_static_channels,
            self.n_chans_full,
        )
        amp_vecs = torch.as_tensor(amp_vecs)
        if device is not None:
            amp_vecs = amp_vecs.to(device)
        return amp_vecs


default_fa_kwargs = dict(
    latent_update="gradient",
    do_prior=False,
)

default_fa_fit_kwargs = dict(
    lr=0.05,
    eps=1e-3,
    n_iter=200,
    loss_converged=1e-2,
)

default_residual_pca_kwargs = dict(
    centered=False,
    atol=0.1,
    max_iter=25,
    pca_on_waveform_channels=True,
    impute_zeros=False,
    pca_noise_scale=0.0,
)


class InterpUnit(torch.nn.Module):
    """InterpUnit

    This is the middleman between the overall clusterer and the actual models.
    It knows how to deal with the channels the unit lives on, converting wfs
    to those channels before passing them to the model.
    """

    def __init__(
        self,
        t_bounds,
        waveform_rank,
        n_chans_unit,
        n_chans_waveform,
        n_chans_full,
        min_overlap=0.5,
        residual_pca_rank=2,
        do_interp=True,
        amplitude_scaling_std=np.sqrt(0.001),
        amplitude_scaling_limit=1.2,
        fa_kwargs=default_fa_kwargs,
        residual_pca_kwargs=default_residual_pca_kwargs,
        fa_fit_kwargs=default_fa_fit_kwargs,
        channel_strategy="snr",
        channel_strategy_snr_min=25.0,
        channel_strategy_snr_minamp=3.0,
        var_prior_count=5.0,
        var_cov_max=0.05,
        imputation_kind=None,
        impute_before_center=False,
        kriging_sigma=None,
        batch_size=16384,
        impute_batch_size=256,
        pca_on_waveform_channels=True,
        scale_residual_embed=False,
    ):
        super().__init__()
        self.residual_pca_rank = residual_pca_rank
        self.do_interp = do_interp
        self.min_overlap = min_overlap
        self.n_chans_full = n_chans_full
        self.n_chans_unit = n_chans_unit
        self.n_chans_waveform = n_chans_waveform
        self.waveform_rank = waveform_rank
        self.scale_residual_embed = scale_residual_embed
        self.channel_strategy = channel_strategy
        self.channel_strategy_snr_min = channel_strategy_snr_min
        self.channel_strategy_snr_minamp = channel_strategy_snr_minamp
        self.t_bounds = t_bounds
        self.batch_size = batch_size
        self.var_prior_count = var_prior_count
        self.var_cov_max = var_cov_max
        self.imputation_kind = imputation_kind
        self.impute_before_center = impute_before_center
        self.kriging_sigma = kriging_sigma
        self.impute_batch_size = impute_batch_size

        pca_centered = True
        if self.do_interp:
            self.fa_fit_kwargs = fa_fit_kwargs
            pca_centered = False
        self.pca_kwargs = residual_pca_kwargs | dict(
            centered=pca_centered, rank=residual_pca_rank
        )
        self.pca_on_waveform_channels = self.pca_kwargs.pop(
            "pca_on_waveform_channels", True
        )
        self.pca_impute_zeros = self.pca_kwargs.pop("pca_impute_zeros", False)
        self.pca_noise_scale = self.pca_kwargs.pop("pca_noise_scale", 0.0)

        # unit channels logic
        self.needs_fit = True
        self.register_buffer(
            "inv_lambda", torch.tensor(1.0 / (amplitude_scaling_std**2))
        )
        self.register_buffer(
            "scale_clip_low", torch.tensor(1.0 / amplitude_scaling_limit)
        )
        self.register_buffer("scale_clip_high", torch.tensor(amplitude_scaling_limit))

    def _init_models(self):
        self.input_dim = self.waveform_rank * self.n_chans_unit
        if self.do_interp:
            self.interp = InterpFactorAnalysis(
                self.t_bounds, output_dim=self.input_dim, **fa_kwargs
            )

        pca_input_dim = self.input_dim
        if self.pca_on_waveform_channels:
            pca_input_dim = self.waveform_rank * n_chans_waveform

        self.pca = MaskedPCA(input_dim=pca_input_dim, **self.pca_kwargs)

    def _needs_to_be_fitted(self):
        assert not self.needs_fit

    def determine_position_(
        self,
        waveforms,
        waveform_channels,
        padded_geom,
        cluster_channel_index,
        channels=None,
        max_channel=None,
    ):
        if cluster_channel_index is not None:
            assert cluster_channel_index.shape == (self.n_chans_full, self.n_chans_unit)

        device = waveforms.device
        if channels is None:
            geom = padded_geom[:-1]
            norms = torch.linalg.norm(waveforms, dim=1)
            ones = torch.isfinite(norms).to(norms)
            (valid_ix,) = torch.nonzero(ones.view(-1), as_tuple=True)
            padded_count = torch.zeros(len(padded_geom), device=norms.device)
            spiketorch.add_at_(padded_count, waveform_channels.view(-1), ones.view(-1))
            padded_ampmean = torch.zeros(len(padded_geom), device=norms.device)
            spiketorch.reduce_at_(padded_ampmean, waveform_channels.view(-1)[valid_ix], norms.view(-1)[valid_ix], reduce="mean", include_self=False)
            count = padded_count[:-1]
            ampmean = padded_ampmean[:-1]
            snr = ampmean * torch.sqrt(count)
            self.snr = snr
            self.count = count
            self.com = (snr * geom[:, 1]).sum() / snr.sum()
            if self.channel_strategy in ("snr", "peak"):
                max_channel = snr.argmax()
            elif self.channel_strategy in ("com",):
                fullcom = (snr[:, None] * geom).sum() / snr.sum()
                max_channel = (geom - fullcom).square().sum(1).argmin()
            else:
                assert False

            if self.channel_strategy in ("peak", "com"):
                channels = cluster_channel_index[max_channel]
            else:
                mask = torch.logical_and(
                    snr > self.channel_strategy_snr_min,
                    ampmean > self.channel_strategy_snr_minamp,
                )
                (channels,) = torch.nonzero(mask, as_tuple=True)
        else:
            channels = channels.clone()
            max_channel = max_channel.clone()

        self.n_chans_unit = channels.numel()
        channel_reindexer = torch.full(
            (self.n_chans_full + 1,), self.n_chans_unit, device=device
        )
        my_valid = channels < self.n_chans_full
        my_ixs = torch.arange(self.n_chans_unit, device=device)[my_valid]
        channel_reindexer[channels[my_valid]] = my_ixs
        if hasattr(self, "channel_reindexer"):
            self.channel_reindexer.copy_(channel_reindexer)
            self.max_channel.copy_(max_channel)
        else:
            self.register_buffer("channel_reindexer", channel_reindexer)
            self.register_buffer("max_channel", max_channel)
        self.register_buffer("channels", channels)
        self.register_buffer("channels_valid", channels[my_valid])

    def overlaps(self, static_channels):
        """
        Arguments
        ---------
        static_channels : (n_spikes, n_chans_wf)

        Returns
        -------
        overlaps : (n_spikes)
        rel_ix : (n_spikes, n_chans_unit)
        """
        rel_ix = self.rel_ix(static_channels)
        overlap_num = (rel_ix < self.n_chans_unit).sum(1)
        overlap_den = (static_channels < self.n_chans_full).sum(1)
        overlaps = overlap_num / overlap_den
        return overlaps, rel_ix

    def rel_ix(self, static_channels):
        """
        Arguments
        ---------
        static_channels : (n_spikes, n_chans_wf)

        Returns
        -------
        rel_ix : (n_spikes, n_chans_unit)
        """
        rel_ix = torch.take_along_dim(
            self.channel_reindexer[None],
            static_channels,
            axis=1,
        )
        return rel_ix

    def get_means(self, times=None, n=None, padded=False, constant_value=torch.nan):
        single = times is None and n is None
        if not single and n is None:
            n = len(times)
        if self.do_interp:
            assert not single
            _, means_flat = self.interp(times)
            means = means_flat.view(n, self.waveform_rank, self.n_chans_unit)
            if padded:
                means = F.pad(means, (0, 1), value=constant_value)
        else:
            mean_flat = self.mean
            means = mean_flat.view(self.waveform_rank, self.n_chans_unit)
            if padded:
                means = F.pad(means, (0, 1), value=constant_value)
            if not single:
                means = means[None].broadcast_to(n, *means.shape).contiguous()

        return means

    def to_unit_channels(
        self,
        waveforms,
        times=None,
        waveform_channels=None,
        rel_ix=None,
        fill_mode="mean",
        constant_value=torch.nan,
    ):
        """Shift waveform from its channels to this unit's

        How to fill in the blanks? fill_mode="mean" uses this unit's possibly time-varying
        mean. fill_mode="constant" uses constant_value.
        """
        if rel_ix is None:
            rel_ix = self.rel_ix(waveform_channels)

        if torch.is_tensor(fill_mode):
            waveforms_rel = fill_mode
        elif fill_mode == "mean":
            waveforms_rel = self.get_means(
                times, n=len(rel_ix), padded=True, constant_value=constant_value
            )
        elif fill_mode == "constant":
            waveforms_rel = torch.full(
                (len(waveforms), self.waveform_rank, self.n_chans_unit + 1),
                constant_value,
                dtype=waveforms.dtype,
                device=waveforms.device,
            )

        n, r, c = waveforms.shape
        rel_ix_scatter = rel_ix[:, None, :].broadcast_to((n, r, rel_ix.shape[-1]))
        waveforms_rel.scatter_(src=waveforms, dim=2, index=rel_ix_scatter)
        return waveforms_rel[..., : self.n_chans_unit]

    def to_waveform_channels(
        self,
        waveforms_rel,
        waveform_channels=None,
        rel_ix=None,
        already_padded=False,
        constant_value=torch.nan,
    ):
        if rel_ix is None:
            rel_ix = self.rel_ix(waveform_channels)

        n = waveforms_rel.shape[0]
        waveforms_rel = waveforms_rel.reshape(n, -1, self.n_chans_unit + already_padded)
        if not already_padded:
            waveforms_rel = F.pad(waveforms_rel, (0, 1), value=constant_value)
        rel_ix = rel_ix[:, None, :].broadcast_to(
            (n, self.waveform_rank, rel_ix.shape[-1])
        )
        return torch.gather(waveforms_rel, dim=2, index=rel_ix)

    def residuals_rel(
        self,
        times,
        waveforms,
        waveform_channels,
        waveform_channel_index=None,
        rel_ix=None,
        batch_size=None,
        out=None,
        embed=False,
        padded=False,
        fill_mode="mean",
        constant_value=torch.nan,
    ):
        if batch_size is None:
            batch_size = self.batch_size
        if rel_ix is None:
            rel_ix = self.rel_ix(waveform_channels)
        n = len(waveforms)
        if embed:
            out_shape = (self.residual_pca_rank,)
        else:
            out_shape = (self.waveform_rank, self.n_chans_unit)
        if out is None:
            out = torch.empty(
                (n, *out_shape),
                dtype=waveforms.dtype,
                device=waveforms.device,
            )
        if embed and self.pca_on_waveform_channels:
            wfc = waveform_channel_index[self.max_channel][None]
        for j in range(0, n, batch_size):
            sl = slice(j, min(j + batch_size, n))
            means_rel = self.get_means(times[sl], padded=True)
            if fill_mode == "mean":
                fill_kw = dict(fill_mode=means_rel.clone().detach())
            else:
                fill_kw = dict(fill_mode=fill_mode, constant_value=constant_value)
            resids = self.to_unit_channels(
                waveforms=waveforms[sl],
                rel_ix=rel_ix[sl],
                **fill_kw,
            )
            means_rel = means_rel[..., :-1]
            if self.scale_residual_embed:
                means_rel = means_rel.mul(
                    self.get_scalings(resids, means_rel)[:, None, None]
                )
            resids.sub_(means_rel)
            if embed:
                if self.pca_on_waveform_channels:
                    resids = self.to_waveform_channels(
                        resids, wfc.broadcast_to((len(resids), *wfc.shape))
                    )
                resids = resids.reshape(len(resids), -1)
                out[sl] = self.pca.transform_precentered(resids)
            else:
                out[sl] = resids

        return out

    def residual_embed(
        self,
        times,
        waveforms,
        waveform_channels,
        waveform_channel_index=None,
        rel_ix=None,
        batch_size=None,
        out=None,
    ):
        return self.residuals_rel(
            times,
            waveforms,
            waveform_channels,
            waveform_channel_index=waveform_channel_index,
            rel_ix=rel_ix,
            batch_size=batch_size,
            out=out,
            embed=True,
        )

    def get_scalings(self, waveforms, means, already_masked=False, scaled=True):
        if not already_masked:
            mask = torch.isfinite(waveforms[:, 0, :]).unsqueeze(1).to(waveforms)
            waveforms = torch.nan_to_num(waveforms)
            means = torch.nan_to_num(means * mask)
        dots = means.mul(waveforms).sum(dim=(1, 2))
        recons_sumsq = means.square().sum(dim=(1, 2))
        scalings = (dots + self.inv_lambda).div_(recons_sumsq + self.inv_lambda)
        scalings = scalings.clip_(self.scale_clip_low, self.scale_clip_high)
        return scalings

    def spike_badnesses(
        self,
        times,
        waveforms,
        waveform_channels,
        kinds=("l2", "1-r^2", "1-scaledr^2"),
        overlaps=None,
        rel_ix=None,
        common_chans=False,
        min_overlap=None,
    ):
        """Badnesses

        How bad is this unit at explaining waveforms on their channels?
        """
        if min_overlap is None:
            min_overlap = self.min_overlap
        # a client may already know exactly which spikes they want to compare
        spike_ix = slice(None)
        if rel_ix is None:
            overlaps, rel_ix = self.overlaps(waveform_channels)
            (spike_ix,) = torch.nonzero(overlaps >= min_overlap, as_tuple=True)
            waveforms = waveforms[spike_ix]
            rel_ix = rel_ix[spike_ix]
            overlaps = overlaps[spike_ix]
            times = times[spike_ix]

        # masks are all or nothing along axis=1
        # this mask says: which of each waveform's channels is one I have got?
        maskb = (rel_ix < self.n_chans_unit).unsqueeze(1)
        mask = maskb.to(waveforms)
        if common_chans:
            mask *= torch.isfinite(waveforms[:, :1]).to(mask)
            waveforms = torch.nan_to_num(waveforms * mask)
        else:
            waveforms = torch.nan_to_num(waveforms)

        # try to reconstruct spikes
        recons_rel = self.get_means(times, padded=True)
        if times is None:
            recons_rel = recons_rel[None]
        recons = self.to_waveform_channels(
            recons_rel, rel_ix=rel_ix, already_padded=True
        )
        recons = torch.nan_to_num(recons * mask)

        badnesses = {}
        if "l2" in kinds or "1-r^2" in kinds:
            l2 = waveforms.sub(recons).square().sum(dim=(1, 2))
        if "mse" in kinds:
            badnesses["mse"] = waveforms.sub(recons).square().mean(dim=(1, 2))
        if "l2" in kinds:
            badnesses["l2"] = l2
        if any("r^2" in k for k in kinds):
            wf_l2 = waveforms.square().sum(dim=(1, 2))
        if "1-r^2" in kinds:
            badnesses["1-r^2"] = l2 / wf_l2
        if "scaledl2" or "1-scaledr^2" in kinds:
            scalings = self.get_scalings(waveforms, recons, already_masked=True)
            scaled_l2 = (
                waveforms.sub(scalings[:, None, None] * recons).square().sum(dim=(1, 2))
            )
        if "scaledl2" in kinds:
            badnesses["scaledl2"] = scaled_l2
        if "1-scaledr^2" in kinds:
            badnesses["1-scaledr^2"] = scaled_l2 / wf_l2
        if "1-maxr^2" in kinds:
            maxsq = torch.amax(waveforms.sub(recons).square(), dim=(1, 2))
            meansq = waveforms.square().sum(dim=1).max(dim=1).values
            badnesses["1-maxr^2"] = maxsq / meansq
        if "diagz" in kinds:
            dx = waveforms.sub(recons)
            var = self.var.view(self.waveform_rank, self.n_chans_unit)
            var = var[None].broadcast_to((len(dx), *var.shape))
            var = self.to_waveform_channels(var, rel_ix=rel_ix)
            var = torch.where(maskb, var, 1.0)
            mah = (dx.square() / var).mean(1).sum(1)
            dof = mask.sum(dim=(1, 2)).to(mah)
            badnesses["diagz"] = mah / dof

        return spike_ix, overlaps, badnesses

    def divergence(
        self,
        other,
        kind="1-r^2",
        aggregate=torch.amax,
        min_overlap=None,
        subset_channel_index=None,
        common_chans=False,
    ):
        """Try to explain other units' mean (traces)"""
        if min_overlap is None:
            min_overlap = self.min_overlap

        other_channels = other.channels
        if subset_channel_index is not None:
            other_channels = subset_channel_index[other.max_channel]

        overlaps, rel_ix = self.overlaps(other_channels[None])
        if overlaps.squeeze() < min_overlap:
            return torch.inf

        if not self.do_interp:
            # simple static case
            other_waveform = other.get_means()[None]
            if subset_channel_index is not None:
                other_waveform = other.to_waveform_channels(
                    other_waveform,
                    waveform_channels=other_channels[None],
                )
            _, _, badnesses = self.spike_badnesses(
                times=None,
                waveforms=other_waveform,
                waveform_channels=other_channels[None],
                kinds=(kind,),
                overlaps=overlaps,
                min_overlap=min_overlap,
                rel_ix=rel_ix,
                common_chans=common_chans,
            )
            div = badnesses[kind].squeeze()
            return div

        assert not common_chans

        # find grid times in common
        common_grid = torch.logical_and(
            self.interp.grid_fitted,
            other.interp.grid_fitted,
        )
        if not common_grid.any():
            return torch.inf
        common_times = self.interp.grid[common_grid]
        nt = len(common_times)

        # compare
        other_waveforms = other.get_means(common_times)
        if subset_channel_index is not None:
            other_waveforms = other.to_waveform_channels(
                other_waveforms,
                waveform_channels=other_channels[None].broadcast_to(
                    (nt, *other_channels.shape)
                ),
            )
        _, _, badnesses = self.spike_badnesses(
            times=common_times,
            waveforms=other_waveforms,
            waveform_channels=other_channels[None],
            kinds=(kind,),
            overlaps=overlaps,
            rel_ix=rel_ix,
        )
        div = badnesses[kind].squeeze()

        # aggregate over time
        div = aggregate(div)

        return div

    def self_divergence_matrix(self):
        pass

    @torch.enable_grad()
    def fit_center(
        self,
        times,
        waveforms,
        waveform_channels,
        geom,
        padded_geom=None,
        cluster_channel_index=None,
        waveform_channel_index=None,
        show_progress=False,
        with_var=True,
        weights=None,
        channels=None,
        max_channel=None,
    ):
        # transfer waveform -> unit channels, filling with nans
        self.train()
        self.determine_position_(
            waveforms,
            waveform_channels,
            padded_geom,
            cluster_channel_index,
            channels=channels,
            max_channel=max_channel,
        )
        if not self.n_chans_unit:
            raise ValueError("Low-signal unit.")
        self._init_models()
        self.to(waveforms.device)
        n = len(times)
        assert n > 0
        if self.impute_before_center:
            waveforms_rel = self.impute(
                times,
                waveforms,
                waveform_channels,
                padded_registered_geom=padded_geom,
                centered=False,
            )
        else:
            waveforms_rel = self.to_unit_channels(
                waveforms,
                times,
                waveform_channels=waveform_channels,
                fill_mode="constant",
                constant_value=torch.nan,
            )

        # fit/transform with the interpolator
        if self.do_interp:
            self.interp.fit(
                times,
                waveforms_rel.reshape(n, -1),
                show_progress=show_progress,
                **self.fa_fit_kwargs,
            )
            assert not with_var
            assert weights is None
        else:
            if weights is None:
                weights = torch.ones(n).to(waveforms)
            weights = weights / weights.sum()
            mean = torch.nansum(weights[:, None] * waveforms_rel.reshape(n, -1), dim=0)
            mean = torch.nan_to_num(mean)
            if mean.abs().max() > 1000:
                print(f"{torch.nan_to_num(waveforms).min()=} {torch.nan_to_num(waveforms).max()=}")
                print(f"{waveforms_rel.min()=} {waveforms_rel.max()=}")
                print(f"{mean.min()=} {mean.max()=}")
                print(f"{weights.min()=} {weights.max()=}")
                print(f"{waveforms.shape=} {waveforms_rel.shape=}")
                print("!!!!!")
                print(f"{padded_geom[self.channels]=}")
                raise ValueError("gah")
            if hasattr(self, "mean"):
                self.mean.copy_(mean)
            else:
                self.register_buffer(
                    "mean",
                    mean,
                )
            if with_var:
                dxsq = (waveforms_rel.reshape(n, -1) - self.mean).square_()
                var = torch.nansum(weights[:, None] * dxsq, dim=0)
                var = torch.nan_to_num(var, nan=1.0)
                nobs = torch.isfinite(waveforms_rel).sum(0).view(-1).to(var)
                lambd = nobs / (nobs + self.var_prior_count)
                var = var * lambd + 1.0 * (1 - lambd)
                std = var.sqrt()

                # soft version of min
                norm_mean = torch.norm(self.mean)
                cov = std / norm_mean
                cov = cov.clamp(max=self.var_cov_max)
                var = torch.square(cov * norm_mean)

                # var = 1.0 + F.softplus(var - 1.0)
                if hasattr(self, "var"):
                    self.var.copy_(var)
                else:
                    self.register_buffer(
                        "var",
                        var,
                    )

    def impute(
        self,
        times,
        waveforms,
        waveform_channels,
        waveform_channel_index=None,
        imputation_kind=None,
        padded_registered_geom=None,
        centered=True,
        valid_only=False,
    ):
        n = len(times)
        imputation_kind = self.imputation_kind

        if imputation_kind == "pca":
            assert not self.needs_fit
            assert centered
            residuals = self.residuals_rel(
                times,
                waveforms,
                waveform_channels,
                padded=False,
                fill_mode="mean" if imputation_kind == "pca" else "constant",
            )
            residuals = residuals.reshape(n, -1)
            if self.pca_on_waveform_channels:
                wfcs = waveform_channel_index[self.max_channel]
                wfcs = wfcs[None].broadcast_to((len(residuals), *wfcs.shape))
                residuals = self.to_waveform_channels(residuals, wfcs)
            embeds = self.pca.transform_precentered(residuals)
            recons = self.pca.backward_precentered(embeds)
            imputed = torch.where(
                torch.isfinite(residuals.view(n, -1)),
                residuals.view(n, -1),
                recons.view(n, -1),
            )
        elif imputation_kind.startswith("kriging"):
            assert not valid_only
            residuals = residuals.view((n, self.waveform_rank, -1))
            assert not self.pca_on_waveform_channels, "Could support this with target kernel"
            source_pos = padded_registered_geom[self.channels]
            source_kernel = torch.cdist(source_pos, source_pos)
            source_kernel = source_kernel.square_().mul_(-1.0 / (2 * self.kriging_sigma**2))
            torch.nan_to_num(source_kernel, nan=-torch.inf, out=source_kernel)
            source_kernel = source_kernel.exp_()
            source_inv = torch.linalg.pinv(source_kernel)
            source_pos = source_pos[None].broadcast_to((n, *source_pos.shape))
            source_inv = source_inv[None].broadcast_to((n, *source_inv.shape))
            target_pos = padded_registered_geom[self.channels]
            target_pos = target_pos[None].broadcast_to((n, *target_pos.shape))
            recons = kernel_interpolate(
                residuals,
                source_pos,
                target_pos,
                source_kernel_invs=source_inv,
                sigma=self.kriging_sigma,
                kind=imputation_kind,
            )
            imputed = torch.where(
                torch.isfinite(residuals.view(n, -1)),
                residuals.view(n, -1),
                recons.view(n, -1),
            )
        elif imputation_kind in ("kernel", "normalized"):
            # source_pos = padded_registered_geom[waveform_channels]
            c = self.channels_valid if valid_only else self.channels
            target_pos = padded_registered_geom[c]
            # source_pos = source_pos[None].broadcast_to((n, *source_pos.shape))
            target_pos = target_pos[None].broadcast_to((n, *target_pos.shape))
            imputed = torch.empty((*waveforms.shape[:2], c.numel()), dtype=waveforms.dtype, device=waveforms.device)
            for j in range(0, n, self.impute_batch_size):
                sl = slice(j, j + self.impute_batch_size)
                kernel_interpolate(
                    waveforms[sl],
                    padded_registered_geom[waveform_channels[sl]],
                    target_pos[sl],
                    sigma=self.kriging_sigma,
                    kind=imputation_kind,
                    out=imputed[sl],
                )
            imputed = imputed.reshape(n, -1)
            if centered:
                imputed -= self.mean
        else:
            assert False
        return imputed

    def fit_residual(
        self,
        times,
        waveforms,
        waveform_channels,
        geom,
        waveform_channel_index,
        show_progress=False,
        imputation_kind=None,
        padded_registered_geom=None,
    ):
        rel_ix = self.rel_ix(waveform_channels)
        n = len(times)
        if self.do_interp:
            assert not self.pca_impute_zeros, "not implemented"
            residuals = self.residuals_rel(
                times,
                waveforms,
                waveform_channels,
                rel_ix=rel_ix,
                padded=False,
            )
        else:
            if imputation_kind is not None and imputation_kind != "pca":
                waveforms_rel = self.impute(
                    times,
                    waveforms,
                    waveform_channels,
                    waveform_channel_index,
                    imputation_kind=imputation_kind,
                    padded_registered_geom=padded_registered_geom,
                )
                residuals = waveforms_rel.reshape(n, -1) - self.mean
            else: 
                waveforms_rel = self.to_unit_channels(
                    waveforms,
                    times,
                    rel_ix=rel_ix,
                    fill_mode="constant",
                    constant_value=0.0 if self.pca_impute_zeros else torch.nan,
                )
                residuals = waveforms_rel.reshape(n, -1) - self.mean
                if self.pca_on_waveform_channels:
                    wfcs = waveform_channel_index[self.max_channel]
                    wfcs = wfcs[None].broadcast_to((len(residuals), *wfcs.shape))
                    residuals = self.to_waveform_channels(residuals, wfcs)
        # if self.pca_impute_zeros:
        #     torch.nan_to_num(residuals, out=residuals)
        if self.pca_noise_scale:
            residuals = torch.normal(residuals, std=self.pca_noise_scale)
        residuals = residuals.reshape(len(residuals), -1)
        self.pca.fit(residuals)
        self.needs_fit = False
        self.eval()


@dataclasses.dataclass
class DPCSplitKwargs:
    rank: int = 2
    sigma_local: Union[str, float] = "rule_of_thumb"
    sigma_regional: Optional[float] = None
    n_neighbors_search: int = 250
    allow_single_cluster_outlier_removal: bool = True
    recursive: bool = True
    split_on_train: bool = False
    radius_search: float = 5.0
    reassign_within_split: bool = False


@dataclasses.dataclass
class ContinuitySplitKwargs:
    threshold: float = 0.25
    scaled: bool = True


class InterpClusterer(torch.nn.Module):
    """Mixture of InterpUnits."""

    def __init__(
        self,
        sorting,
        t_bounds,
        motion_est=None,
        fit_radius=35.0,
        waveform_radius=25.0,
        min_overlap=0.5,
        residual_pca_rank=2,
        min_cluster_size=50,
        n_spikes_fit=2048,
        do_interp=False,
        fa_kwargs=default_fa_kwargs,
        residual_pca_kwargs=default_residual_pca_kwargs,
        scale_residual_embed=False,
        dpc_split_kwargs=DPCSplitKwargs(),
        continuity_split_kwargs=ContinuitySplitKwargs(),
        sampling_method: str = "random",
        split_sampling_method: str = "time_amp_reweighted",
        split_waveform_kind: str = "original",
        in_memory=True,
        keep_spikes="byamp",
        max_n_spikes=5000000,
        reassign_metric="1-scaledr^2",
        merge_metric="1-scaledr^2",
        merge_threshold=0.25,
        zip_threshold=0.15,
        merge_sym_function=torch.maximum,
        merge_linkage="complete",
        merge_with_bimodality=True,
        bimodality_threshold=0.5,
        channel_strategy="snr",
        channel_strategy_snr_min=25.0,
        channel_strategy_snr_minamp=2.5,
        merge_on_waveform_radius=True,
        interp_kind="nearest",
        imputation_kind="pca",
        impute_before_center=False,
        drift_positions="channel",
        match_threshold=1.0,
        sampling_sigma=0.5,
        label_dtype=torch.int32,
        clustering_config: ClusteringConfig = None,
        load_tpca=True,
        on_device=False,
        pin=False,
        batch_size=16384,
        tpca_feature_name="collisioncleaned_tpca_features",
        kriging_sigma="pitch",
        rg=0,
    ):
        super().__init__()
        self.min_cluster_size = min_cluster_size
        self.n_spikes_fit = n_spikes_fit
        self.rg = np.random.default_rng(rg)
        self.t_bounds = t_bounds
        self.clustering_config = clustering_config
        self.do_interp = do_interp
        self.sampling_method = sampling_method
        self.split_sampling_method = split_sampling_method
        self.split_waveform_kind = split_waveform_kind
        self.sampling_sigma = sampling_sigma
        self.zip_threshold = zip_threshold
        self.channel_strategy = channel_strategy
        self.batch_size = batch_size
        self.merge_with_bimodality = merge_with_bimodality
        self.bimodality_threshold = bimodality_threshold
        self._reas_bufs = {}
        self.labels_lock = threading.Lock()
        self.imputation_kind = imputation_kind

        self.data = _load_data(
            sorting,
            motion_est,
            fit_radius,
            reassign_wf_radius=waveform_radius,
            in_memory=in_memory,
            keep=keep_spikes,
            max_n_spikes=max_n_spikes,
            load_tpca=load_tpca,
            on_device=on_device,
            pin=pin,
            rg=self.rg,
            tpca_feature_name=tpca_feature_name,
            interp_kind=interp_kind,
            drift_positions=drift_positions,
            kriging_sigma=kriging_sigma,
        )
        self.residual_pca_rank = residual_pca_rank
        self.unit_kw = dict(
            t_bounds=t_bounds,
            n_chans_unit=self.data.n_chans_unit if channel_strategy != "snr" else None,
            n_chans_waveform=self.data.n_chans_waveform,
            waveform_rank=self.data.waveform_rank,
            min_overlap=min_overlap,
            residual_pca_rank=residual_pca_rank,
            fa_kwargs=fa_kwargs,
            residual_pca_kwargs=residual_pca_kwargs,
            n_chans_full=self.data.n_chans_full,
            scale_residual_embed=scale_residual_embed,
            channel_strategy=channel_strategy,
            channel_strategy_snr_min=channel_strategy_snr_min,
            channel_strategy_snr_minamp=channel_strategy_snr_minamp,
            kriging_sigma=self.data.kriging_sigma,
            imputation_kind=imputation_kind,
            impute_before_center=impute_before_center,
        )

        torch.manual_seed(self.rg.bit_generator.random_raw())
        self.labels = torch.tensor(
            self.data.spike_train.labels[self.data.keepers], dtype=label_dtype
        )
        self.models = torch.nn.ModuleDict()
        self.register_buffer("_device", torch.tensor(0))
        self.dpc_split_kw = dpc_split_kwargs
        self.continuity_split_kwargs = continuity_split_kwargs

        self.reassign_metric = reassign_metric
        self.merge_metric = merge_metric
        self.merge_threshold = merge_threshold
        self.merge_linkage = merge_linkage
        self.min_overlap = min_overlap
        self.merge_sym_function = merge_sym_function
        self.merge_on_waveform_radius = merge_on_waveform_radius
        self.match_threshold = match_threshold

        self.cleanup()

        # self.m_step()
        # self.order_by_depth()

    @staticmethod
    def normalize_key(ix):
        if torch.is_tensor(ix):
            ix = ix.item()
        if isinstance(ix, np.ndarray):
            ix = ix.item()
        return str(ix)

    def __getitem__(self, ix):
        if not ix in self:
            raise KeyError(f"{ix} (normalized: {self.normalize_key(ix)})")
        ix = self.normalize_key(ix)
        return self.models[ix]

    def __setitem__(self, ix, value):
        ix = self.normalize_key(ix)
        self.models[ix] = value

    def update(self, other):
        for key, value in other.items():
            self[key] = value

    def __contains__(self, ix):
        ix = self.normalize_key(ix)
        return ix in self.models

    def unit_ids(self):
        ids = torch.unique(self.labels)
        ids = ids[ids >= 0]
        return ids

    @property
    def device(self):
        return self._device.device

    def update_labels(self, old_labels, new_labels=None, flat=False):
        """
        Re-label units. This should not split labels, but merging is OK.

        Arguments
        ---------
        old_labels : (n_units,)
        new_labels : (n_units,)
        """
        # invariant:
        # labels in self.labels **always** match keys in self.models
        # this means that things which change the meaning of labels
        # must update both structures
        if new_labels is None:
            new_labels = torch.arange(len(old_labels))

        if self.models:
            new_models = {}
            order = torch.argsort(new_labels)
            for j in order:
                oldk = self.normalize_key(old_labels[j])
                newk = self.normalize_key(new_labels[j])
                if newk not in new_models and oldk in self:
                    new_models[newk] = self[oldk]
            self.models.clear()
            self.update(new_models)

        if flat:
            kept = self.labels >= 0
            label_indices = self.labels[kept]
        else:
            kept = torch.isin(self.labels, old_labels)
            label_indices = torch.searchsorted(old_labels, self.labels[kept])
        self.labels[kept] = new_labels.to(self.labels.dtype)[label_indices]
        self.labels[torch.logical_not(kept)] = -1

    def cleanup(self, min_cluster_size=None, remove_low_snr=False):
        """Remove small units and make labels contiguous."""
        old_labels, counts = torch.unique(self.labels, return_counts=True)
        counts = counts[old_labels >= 0]
        old_labels = old_labels[old_labels >= 0]

        if min_cluster_size is None:
            min_cluster_size = self.min_cluster_size

        big_enough = counts >= min_cluster_size
        if remove_low_snr and self.channel_strategy == "snr" and len(self.models):
            nc = [self[u].n_chans_unit for u in self.unit_ids()]
            min_chans = self.data.n_chans_reassign * self.min_overlap
            big_enough = torch.logical_and(big_enough, torch.tensor(nc) >= min_chans)

        n_removed = torch.logical_not(big_enough).sum()
        if n_removed:
            count_removed = counts[torch.logical_not(big_enough)].sum()
            pct_removed = 100 * count_removed / len(self.data.keepers)
            print(
                f"Removed {n_removed} too-small units ({pct_removed:0.1f}% of spikes). "
                f"New unit count: {big_enough.sum()}."
            )
        self.update_labels(old_labels[big_enough], flat=False)
        return big_enough

    def order_by_depth(self):
        """Reorder labels by unit CoM depth."""
        if not self.models:
            return

        unit_ids = self.unit_ids()
        order = np.argsort([self[uid].com.numpy(force=True) for uid in unit_ids])
        # this second argsort never fails to impress (my brain into a small cube)
        self.update_labels(
            unit_ids,
            torch.argsort(torch.from_numpy(order)),
            flat=True,
        )

    def m_step(
        self,
        force=True,
        to_fit=None,
        fit_residual=True,
        show_progress=True,
        store=True,
        n_threads=0,
        with_var=True,
        weights_kind=None,
        weights_sparse=None,
        divergences=None,
    ):
        """Fit all models that need fitting."""
        if to_fit is None:
            if force:
                to_fit = self.unit_ids()
            else:
                to_fit = []
                for uid in self.unit_ids():
                    if uid not in self or self[uid].needs_fit:
                        to_fit.append(uid)

        if weights_kind is not None and weights_sparse is None:
            weights_sparse = self.reassignment_weights(
                unit_ids=to_fit,
                show_progress=False,
                n_threads=n_threads,
                kind=weights_kind,
                divergences=divergences,
            )

        if force:
            self.models.clear()

        def fit_unit(j, uid):
            if force or uid not in self:
                model = InterpUnit(do_interp=self.do_interp, **self.unit_kw)
                model.to(self.device)
                if store:
                    self[uid] = model
            else:
                model = self[uid]

            in_unit, train_data = self.get_training_data(
                uid, waveform_kind="original", sampling_method=self.sampling_method
            )

            weights = None
            if weights_sparse is not None:
                weights = (
                    torch.index_select(
                        weights_sparse[j],
                        0,
                        in_unit.to(weights_sparse.device),
                    )
                    .to_dense()
                    .to(train_data["waveforms"])
                )

            try:
                model.fit_center(
                    **train_data, padded_geom=self.data.padded_registered_geom, weights=weights, show_progress=False, with_var=with_var
                )
            except ValueError:
                with self.labels_lock:
                    self.labels[self.labels == uid] = -1
                    if store:
                        del self.models[self.normalize_key(uid)]
                    return
            model.fit_indices = None
            if fit_residual:
                in_unit, train_data = self.get_training_data(
                    uid,
                    waveform_kind=self.split_waveform_kind,
                    sampling_method=self.split_sampling_method,
                )
                if self.channel_strategy != "snr":
                    del train_data["cluster_channel_index"]
                model.fit_residual(
                    **train_data,
                    show_progress=False,
                    imputation_kind=self.imputation_kind,
                    padded_registered_geom=self.data.padded_registered_geom,
                )
                model.fit_indices = in_unit
            return model

        fit_units = []
        if n_threads:
            res = joblib.Parallel(
                n_jobs=n_threads, backend="threading", return_as="generator"
            )(joblib.delayed(fit_unit)(j, uu) for j, uu in enumerate(to_fit))
            if show_progress:
                res = tqdm(res, desc="M step", total=len(to_fit), **tqdm_kw)
            for model in res:
                if model is not None:
                    fit_units.append(model)
        else:
            if show_progress:
                to_fit = tqdm(to_fit, desc="M step", **tqdm_kw)

            for j, uu in enumerate(to_fit):
                fit_units.append(fit_unit(j, uu))

        return fit_units

    def recluster_outliers(self):
        # -2s mark for no re-clustering
        labels_to_cluster = np.full_like(self.data.spike_train.labels, -2)
        labels_to_cluster[self.data.keepers] = -1
        labels_to_cluster[self.data.keepers[self.labels >= 0]] = -2

        reclustered = initial.initial_clustering(
            recording=None,
            sorting=dataclasses.replace(
                self.data.spike_train, labels=labels_to_cluster
            ),
            clustering_config=self.clustering_config,
            motion_est=self.data.motion_est,
        )
        assert np.isin(
            np.flatnonzero(reclustered.labels >= 0),
            np.flatnonzero(labels_to_cluster == -1),
        ).all()
        new_clusters, new_counts = np.unique(reclustered.labels, return_counts=True)
        valid = new_clusters >= 0
        new_counts = new_counts[valid]
        new_clusters = new_clusters[valid]
        orig_outlier_count = (self.labels < 0).sum()
        newly_clustered_count = new_counts.sum()
        new_outlier_count = orig_outlier_count - newly_clustered_count
        orig_outlier_pct = 100 * orig_outlier_count / len(self.data.keepers)
        new_outlier_pct = 100 * new_outlier_count / len(self.data.keepers)
        print(
            f"Reclustering found {new_clusters.size} new clusters with "
            f"spike counts from {new_counts.min()} to {new_counts.max()}. "
            f"Outlier fraction: {orig_outlier_pct:0.1f}% -> {new_outlier_pct:0.1f}%."
        )

        ixs_to_replace = self.labels < 0
        ixs_to_grab = self.data.keepers[self.labels < 0]
        label_start = self.labels.max() + 1
        replacers = torch.as_tensor(
            reclustered.labels[ixs_to_grab],
            dtype=self.labels.dtype,
            device=self.labels.device,
        )
        replacers = torch.where(
            replacers >= 0,
            label_start + replacers,
            -1,
        )
        self.labels[ixs_to_replace] = replacers
        # self.cleanup()

    def residual_dpc_split(self, unit_ids_to_split=None, n_threads=0):
        if unit_ids_to_split is None:
            unit_ids_to_split = list(self.unit_ids())
        n_orig = len(unit_ids_to_split)
        n_splits = []

        @joblib.delayed
        def job(unit_id):
            return self.dpc_split_unit(unit_id)

        pool = joblib.Parallel(
            n_jobs=max(1, n_threads),
            backend="threading",
            return_as="generator",
        )

        while unit_ids_to_split:
            next_ids_to_split = []
            ns = 0
            results = pool(job(uid) for uid in unit_ids_to_split)
            results = tqdm(
                results,
                total=len(unit_ids_to_split),
                desc=f"Split round {len(n_splits)}",
                **tqdm_kw,
            )
            for nnew, new_splits in results:
                next_ids_to_split.extend(new_splits)
                ns += nnew
            n_splits.append(ns)
            self.m_step(fit_residual=False, n_threads=n_threads, show_progress=False)
            self.cleanup(remove_low_snr=True)
            self.m_step(
                fit_residual=True, to_fit=next_ids_to_split, n_threads=n_threads
            )
            unit_ids_to_split = next_ids_to_split
        sequence = "+".join(map(str, n_splits))
        print(f"Split: {n_orig} + ({sequence}) = {len(self.unit_ids())}.")
        # self.cleanup()
        # self.order_by_depth()

    def split_features(self, uid, in_unit=None):
        (in_unit_full,) = (self.labels == uid).nonzero(as_tuple=True)
        if self.dpc_split_kw.split_on_train:
            assert in_unit is None
            if self[uid].pca.train_loadings is None:
                return in_unit_full, None, None
            in_unit = self[uid].fit_indices
            features = self[uid].pca.train_loadings[:, : self.dpc_split_kw.rank]
            features = features.numpy(force=True)
            return in_unit_full, in_unit, features
        if in_unit is None:
            in_unit = in_unit_full
        n = in_unit.numel()
        features = torch.empty((n, self.residual_pca_rank), device=self.device)
        unit = self[uid]
        ci = self.data.registered_original_channel_index
        if self.split_waveform_kind == "reassign":
            ci = self.data.registered_reassign_channel_index
        for sl, data in self.batches(in_unit, waveform_kind=self.split_waveform_kind):
            unit.residual_embed(**data, out=features[sl], waveform_channel_index=ci)
        features = features[:, : self.dpc_split_kw.rank].numpy(force=True)
        return in_unit_full, in_unit, features

    def dpc_split_unit(self, unit_id):
        """
        Updates state by adding new models and updating labels etc after splitting a unit.

        Returns
        -------
        list of new IDs to split
        """
        # invariant: maintains contiguous label space of big-enough units
        unit = self[unit_id]
        in_unit_full, in_unit, features = self.split_features(unit_id)
        if features is None:
            return 0, []
        if in_unit_full.numel() <= self.min_cluster_size:
            return 0, []

        # we may have duplicate features
        features_uniq, inverse = np.unique(features, axis=0, return_inverse=True)

        try:
            split_labels = density.density_peaks_clustering(
                features_uniq,
                sigma_local=self.dpc_split_kw.sigma_local,
                n_neighbors_search=self.dpc_split_kw.n_neighbors_search,
                remove_clusters_smaller_than=self.min_cluster_size,
                radius_search=self.dpc_split_kw.radius_search,
            )
        except ValueError as e:
            print(e)
            return 0, []
        del features

        # handle duplicates
        split_labels_orig = split_labels = split_labels[inverse]

        # -- deal with relabeling
        split_units, counts = np.unique(split_labels, return_counts=True)
        n_split_full = split_units.size
        counts = counts[split_units >= 0]
        assert (counts >= self.min_cluster_size).all()
        split_units = split_units[split_units >= 0]
        n_split = split_units.size

        # case 1: single-unit outlier removal. re-fit but don't re-split.
        if n_split_full - 1 == n_split == 1:
            if self.dpc_split_kw.allow_single_cluster_outlier_removal:
                unit.needs_fit = True
                self.labels[in_unit_full] = -1
                self.labels[in_unit[split_labels >= 0]] = unit_id
                return 0, []

        # case 0: nothing happened.
        if n_split <= n_split_full <= 1:
            return 0, []

        # in all cases below, we want to re-fit this unit
        unit.needs_fit = True

        # case 2: something legitimately took place.
        # here, split_unit 0 retains label unit_id. split units >=1 get new labels.
        assert n_split in (n_split_full, n_split_full - 1)
        assert n_split > 1
        assert split_units[0] == 0

        if not self.dpc_split_kw.reassign_within_split:
            with self.labels_lock:
                self.labels[in_unit_full] = -1
                new_unit_ids = (
                    unit_id,
                    *(
                        self.labels.max()
                        + torch.arange(1, n_split, dtype=self.labels.dtype)
                    ),
                )
                for split_label, new_label in zip(split_units, new_unit_ids):
                    in_split = in_unit[split_labels == split_label]
                    self.labels[in_split] = new_label
            return len(new_unit_ids) - 1, new_unit_ids

        # reassign within units
        new_units = []
        for j, label in enumerate(split_units):
            u = InterpUnit(
                do_interp=False,
                **self.unit_kw,
            )
            inu = in_unit[split_labels == label]
            inu, train_data = self.get_training_data(
                unit_id=None,
                waveform_kind="original",
                in_unit=inu,
                sampling_method=self.sampling_method,
            )
            u.fit_center(**train_data, show_progress=False)
            new_units.append(u)
        divergences = self.reassignment_divergences(
            which_spikes=in_unit_full,
            units=new_units,
            show_progress=False,
            exclude_above=self.match_threshold,
        )
        split_labels = sparse_reassign(divergences)
        kept = np.flatnonzero(split_labels >= 0)

        # if reassign kills everything, just keep the state before reassignment
        if not kept.size:
            with self.labels_lock:
                self.labels[in_unit_full] = -1
                new_unit_ids = (
                    unit_id,
                    *(
                        self.labels.max()
                        + torch.arange(1, n_split, dtype=self.labels.dtype)
                    ),
                )
                for split_label, new_label in zip(split_units, new_unit_ids):
                    in_split = in_unit[split_labels_orig == split_label]
                    self.labels[in_split] = new_label
            return len(new_unit_ids) - 1, new_unit_ids

        split_units, counts = np.unique(split_labels[kept], return_counts=True)
        split_units = split_units[counts >= self.min_cluster_size]
        n_split = split_units.size
        if n_split <= 1:
            return 0, []

        with self.labels_lock:
            self.labels[in_unit_full] = -1
            new_unit_ids = (
                unit_id,
                *(
                    self.labels.max()
                    + torch.arange(1, n_split, dtype=self.labels.dtype)
                ),
            )
            for split_label, new_label in zip(split_units, new_unit_ids):
                in_split = in_unit_full[split_labels == split_label]
                self.labels[in_split] = new_label

        return len(new_unit_ids) - 1, new_unit_ids

    def kmeanspp(
        self,
        unit_id,
        sampling_method="time_amp_reweighted",
        n_clust=5,
        n_iter=0,
        seed_with="mean",
        drop_prop=0.02,
    ):
        in_unit, data = self.get_training_data(
            unit_id,
            n=None,
            in_unit=None,
            waveform_kind=self.split_waveform_kind,
            sampling_method=None,
        )
        n_clust = min(n_clust, in_unit.numel() // self.min_cluster_size)
        waveforms = self[unit_id].impute(
            data["times"],
            data["waveforms"],
            data["waveform_channels"],
            data["waveform_channel_index"],
            imputation_kind=self.imputation_kind,
            padded_registered_geom=self.data.padded_registered_geom,
            valid_only=True,
        )

        # pick centroids and reassign imputed wfs
        n = len(waveforms)
        centroid_ixs = []
        dists = torch.full(
            (n,), torch.inf, dtype=waveforms.dtype, device=waveforms.device
        )
        assignments = torch.zeros((n,), dtype=torch.long, device=self.labels.device)
        err_info = f"{n=} {waveforms.shape=}"
        for j in range(n_clust):
            if j == 0:
                if seed_with == "random":
                    newix = self.rg.integers(n)
                elif seed_with == "mean":
                    newix = (
                        (waveforms - waveforms.mean(0))
                        .square_()
                        .sum(1)
                        .argmax()
                        .numpy(force=True)
                        .item()
                    )
                else:
                    assert False
            else:
                p = torch.nan_to_num(dists)
                p = p / p.sum()
                p = p.numpy(force=True)
                if not np.isclose(p.sum(), 1.0):
                    print(f"bad {p.sum()=}")
                    print(f"{j=}")
                    print(f"{dists=} {dists.min()=} {dists.max()=}")
                    print(f"{dists.shape=} {dists[:5]=}")
                    print(f"{waveforms[0]=}")
                    print(f"{centroid_ixs=}")
                    print(f"{waveforms[centroid_ixs]=}")
                    print(f"{torch.isnan(waveforms).to(torch.float).mean()=}")
                    print(f"{torch.isinf(waveforms).to(torch.float).mean()=}")
                    print(f"{torch.isnan(waveforms[centroid_ixs]).to(torch.float).mean()=}")
                    print(f"{torch.isinf(waveforms[centroid_ixs]).to(torch.float).mean()=}")
                    print(err_info)
                    p = np.full_like(p, 1.0 / p.size)
                newix = self.rg.choice(n, p=p)
            centroid_ixs.append(newix)
            curcent = waveforms[newix][None]
            newdists = (waveforms - curcent).square_().sum(1)
            closer = newdists < dists
            assignments[closer] = j
            dists[closer] = newdists[closer]

        e = None
        if n_iter:
            e = F.one_hot(assignments, num_classes=n_clust).to(waveforms)
            centroids = (e / e.sum(0)).T @ waveforms
            dists = torch.cdist(waveforms, centroids).square_()
            # try:
            for i in range(n_iter):
                # update responsibilities, n x k
                # err_str = ""
                e = F.softmax(-0.5 * dists, dim=1)
                # err_str += "\n" + (f"{i=} aa {e.shape=} {e.sum(0)=} {e.mean(0)=}")
                # err_str += "\n" + (f"{i=} ab {e.shape=} {e.sum(0)=} {e.mean(0)=}")

                # delete too-small centroids
                # err_str += "\n" + (f"{i=} ba {e.shape=} {e.sum(0)=} {e.mean(0)=}")
                if drop_prop is not None:
                    e = e[:, e.mean(0) >= drop_prop]
                # err_str += "\n" + (f"{i=} bb {e.shape=} {e.sum(0)=} {e.mean(0)=}")
                e = e.div_(e.sum(0))
                # err_str += "\n" + (f"{i=} bc {e.shape=} {e.sum(0)=} {e.mean(0)=}")

                # update centroids
                centroids = e.T @ waveforms
                # err_str += "\n" + (f"{i=} {centroids.shape=}")
                # dists = waveforms[:, None, :] - centroids[None, :, :]
                # err_str += "\n" + (f"{i=} {dists.shape=}")
                # dists = dists.square_().sum(2)
                dists = torch.cdist(waveforms, centroids).square_()
                # err_str += "\n" + (f"{i=} {dists.shape=}")
                assignments = torch.argmin(dists, 1)
                if e.shape[1] == 1:
                    break
            # except:
            #     print()
            #     print(f"{torch.isnan(waveforms).to(torch.float).mean()=}")
            #     print(f"{torch.isinf(waveforms).to(torch.float).mean()=}")
            #     centroids = waveforms[centroid_ixs]
            #     print(f"{torch.isnan(centroids).to(torch.float).mean()=}")
            #     print(f"{torch.isinf(centroids).to(torch.float).mean()=}")
            #     print(f"{centroids=}")
            #     print(f"{centroids.min()=}")
            #     print(f"{centroids.max()=}")
            #     print(f"{waveforms.min()=}")
            #     print(f"{waveforms.max()=}")
            #     print(f"{waveforms[10]=}")
            #     print(f"{waveforms[22]=}")
            #     print(f"{waveforms.shape=} {centroids.shape=}")
            #     dists = torch.cdist(waveforms[None], centroids[None])[0]
            #     print(f"yy {dists=} {torch.isnan(dists).any()=} {torch.isinf(dists.any())=}")
            #     print(f"{dists.shape=} {dists.min()=} {dists.max()=}")
            #     print(err_str)
            #     raise

        return in_unit, assignments.to(in_unit), e

    def constrain_split_centroids(
        self, unit_id, in_unit, sub_labels, weights=None, inherit_chans=True
    ):
        ids = torch.unique(sub_labels)
        ids = ids[ids >= 0]
        new_units = []

        # fit sub-units
        chan_kw = {}
        if inherit_chans:
            chan_kw = dict(
                channels=self[unit_id].channels,
                max_channel=self[unit_id].max_channel,
            )
        alive_labels = np.full_like(sub_labels, -1)
        cur_ix = 0
        for j, label in enumerate(ids):
            u = InterpUnit(
                do_interp=False,
                **self.unit_kw,
            )
            inu = in_unit[sub_labels == label]
            w = None
            if weights is not None:
                w = weights[sub_labels == label, j]
            inu, train_data = self.get_training_data(
                unit_id=None,
                waveform_kind="original",
                in_unit=inu,
                sampling_method=self.sampling_method,
            )
            try:
                u.fit_center(**train_data, show_progress=False, padded_geom=self.data.padded_registered_geom, weights=w, **chan_kw)
                new_units.append(u)
                alive_labels[sub_labels == label] = cur_ix
                cur_ix += 1
            except ValueError:
                continue

        # this is alive_label, unit pair
        ju = [(j, u) for j, u in enumerate(new_units) if u.n_chans_unit]

        # sub-merge
        kind = self.merge_metric
        min_overlap = self.min_overlap
        subset_channel_index = None
        if self.merge_on_waveform_radius:
            subset_channel_index = self.data.registered_reassign_channel_index
        nu = len(new_units)
        divergences = torch.full((nu, nu), torch.nan)
        for i, ua in ju:
            for j, ub in ju:
                if i == j:
                    divergences[i, j] = 0
                    continue
                divergences[i, j] = ua.divergence(
                    ub,
                    kind=kind,
                    min_overlap=min_overlap,
                    subset_channel_index=subset_channel_index,
                )

        dists = self.merge_sym_function(divergences, divergences.T)
        dists = dists.numpy(force=True)
        valid = np.isfinite(dists)
        if not valid.all():
            dists[np.logical_not(valid)] = dists[valid].max() + 10
        d = dists[np.triu_indices(dists.shape[0], k=1)]
        Z = linkage(d, method=self.merge_linkage)
        new_labels = fcluster(Z, self.zip_threshold, criterion="distance")
        new_labels -= 1

        _, merge_labels = np.unique(new_labels[alive_labels], return_inverse=True)

        labels = np.where(
            alive_labels >= 0,
            merge_labels,
            -1,
        )
        return labels

    def kmeans_split(self, verbose=False, n_threads=0):
        assert n_threads == 0, "Need to lock rng access"
        n_new = 0
        threaded = bool(n_threads)
        if not threaded:
            for unit_id in tqdm(self.unit_ids(), desc="kmeans split"):
                n_new_unit, newids = self.kmeans_split_unit(unit_id)
                if verbose and newids:
                    print(f"{unit_id=} {newids=}")
                n_new += n_new_unit
        else:
            jobs = self.unit_ids()

            def job(unit_id):
                return self.kmeans_split_unit(unit_id)

            results = joblib.Parallel(
                n_jobs=n_threads,
                backend="threading",
                return_as="generator",
            )(joblib.delayed(job)(u) for u in jobs)
            results = tqdm(results, desc="kmeans split", total=len(jobs))
            for n_new_unit, newids in results:
                if verbose and newids:
                    print(f"{unit_id=} {newids=}")
                n_new += n_new_unit

        print(f"kmeans split broke off {n_new} new units.")

    def kmeans_split_unit(self, unit_id):
        (in_unit_full,) = torch.nonzero(self.labels == unit_id, as_tuple=True)
        in_unit, split_labels, weights = self.kmeanspp(unit_id, n_clust=5, n_iter=50)
        split_units, split_labels = torch.unique(split_labels, return_inverse=True)
        if split_units.numel() <= 1:
            return 0, []
        split_labels_orig = split_labels = self.constrain_split_centroids(
            unit_id, in_unit, split_labels, weights=weights
        )

        split_units, counts = np.unique(split_labels, return_counts=True)
        n_split_full = split_units.size
        valid = np.logical_and(split_units >= 0, counts >= self.min_cluster_size)
        counts = counts[valid]
        split_units = split_units[valid]
        n_split = split_units.size

        # case 1: single-unit outlier removal. re-fit but don't re-split.
        if n_split_full >= n_split == 1:
            if self.dpc_split_kw.allow_single_cluster_outlier_removal:
                self[unit_id].needs_fit = True
                self.labels[in_unit_full] = -1
                self.labels[in_unit[split_labels >= 0]] = unit_id
                return 0, []

        # case 0: nothing happened.
        assert n_split <= n_split_full
        if n_split <= n_split_full <= 1:
            return 0, []

        # in all cases below, we want to re-fit this unit
        self[unit_id].needs_fit = True

        # case 2: something legitimately took place.
        # here, split_unit 0 retains label uid. split units >=1 get new labels.
        assert n_split > 1, f"{n_split=} {n_split_full=}"

        if not self.dpc_split_kw.reassign_within_split:
            with self.labels_lock:
                self.labels[in_unit_full] = -1
                new_unit_ids = (
                    unit_id,
                    *(
                        self.labels.max()
                        + torch.arange(1, n_split, dtype=self.labels.dtype)
                    ),
                )
                for split_label, new_label in zip(split_units, new_unit_ids):
                    in_split = in_unit[split_labels == split_label]
                    self.labels[in_split] = new_label
            return len(new_unit_ids) - 1, new_unit_ids

        # reassign within units
        new_units = []
        for j, label in enumerate(split_units):
            u = InterpUnit(
                do_interp=False,
                **self.unit_kw,
            )
            inu = in_unit[split_labels == label]
            inu, train_data = self.get_training_data(
                unit_id=None,
                waveform_kind="original",
                in_unit=inu,
                sampling_method=self.sampling_method,
            )
            u.fit_center(**train_data, show_progress=False, padded_geom=self.data.padded_registered_geom)
            new_units.append(u)
        divergences = self.reassignment_divergences(
            which_spikes=in_unit_full,
            units=new_units,
            show_progress=False,
            exclude_above=self.match_threshold,
        )
        split_labels = sparse_reassign(divergences)
        kept = np.flatnonzero(split_labels >= 0)

        # if reassign kills everything, just keep the state before reassignment
        if not kept.size:
            with self.labels_lock:
                self.labels[in_unit_full] = -1
                new_unit_ids = (
                    unit_id,
                    *(
                        self.labels.max()
                        + torch.arange(1, n_split, dtype=self.labels.dtype)
                    ),
                )
                for split_label, new_label in zip(split_units, new_unit_ids):
                    in_split = in_unit[split_labels_orig == split_label]
                    self.labels[in_split] = new_label
            return len(new_unit_ids) - 1, new_unit_ids

        split_units, counts = np.unique(split_labels[kept], return_counts=True)
        split_units = split_units[counts >= self.min_cluster_size]
        n_split = split_units.size
        if n_split <= 1:
            return 0, []

        with self.labels_lock:
            self.labels[in_unit_full] = -1
            new_unit_ids = (
                unit_id,
                *(
                    self.labels.max()
                    + torch.arange(1, n_split, dtype=self.labels.dtype)
                ),
            )
            for split_label, new_label in zip(split_units, new_unit_ids):
                in_split = in_unit_full[split_labels == split_label]
                self.labels[in_split] = new_label
        return len(new_unit_ids) - 1, new_unit_ids

    def parcellate(self):
        unit_ids_to_split = list(self.unit_ids())
        n_orig = len(unit_ids_to_split)
        n_splits = 0

        for unit_id in unit_ids_to_split:
            n_splits += self.parcellate_unit(unit_id)

        self.cleanup()

    def parcellate_unit(self, unit_id):
        (in_unit_full,) = torch.nonzero(self.labels == unit_id, as_tuple=True)
        spike_chans = self.data.spike_static_channels[in_unit_full].numpy(force=True)
        unique_neighbs, inverse = np.unique(spike_chans, axis=0, return_inverse=True)
        n_uniq, _ = ushape = unique_neighbs.shape
        if n_uniq == 1:
            return

        denoms = (unique_neighbs < self.data.n_chans_full).sum(1)
        overlaps = np.zeros((n_uniq, n_uniq))
        for i in range(n_uniq):
            target = unique_neighbs[i]
            target = target[target < self.data.n_chans_full]
            iolaps = np.isin(unique_neighbs.ravel(), target).reshape(ushape)
            overlaps[i, :] = iolaps.sum(1) / denoms[i]
        overlaps = np.maximum(overlaps, overlaps.T)

        dists = 1 - overlaps
        d = dists[np.triu_indices(n_uniq, k=1)]
        Z = linkage(d, method="complete")
        threshold = (
            1 - self.min_overlap * self.data.n_chans_waveform / self.data.n_chans_full
        )
        uniq_labels = fcluster(Z, threshold, criterion="distance")

        # regroup
        ids = np.unique(uniq_labels)
        if ids.size == 1:
            return 0

        if unit_id in self:
            self[unit_id].needs_fit = True
        split_labels = uniq_labels[inverse]
        self.labels[in_unit_full] = -1
        new_unit_ids = (
            unit_id,
            *(self.labels.max() + torch.arange(1, ids.size, dtype=self.labels.dtype)),
        )
        for split_label, new_label in zip(ids, new_unit_ids):
            in_split = in_unit_full[split_labels == split_label]
            self.labels[in_split] = new_label
        return ids.size - 1

    def zipper_split(self):
        n_new = 0
        self._zipper_parents = {}
        for unit_id in tqdm(self.unit_ids(), desc="Zipper split"):
            n_new += self.zipper_split_unit(unit_id)
        print(f"Zipper split broke off {n_new} new units.")
        self.m_step()

    def zipper_split_unit(self, unit_id):
        unit = self[unit_id]
        in_unit = in_unit_full = np.flatnonzero(self.labels == unit_id)

        times = self.data.times_seconds[in_unit].numpy(force=True)
        amps = np.nan_to_num(self.data.get_static_amp_vecs(in_unit)).ptp(1)
        z = np.c_[times, amps]
        z /= mad(z, axis=0, keepdims=True)
        split_labels = density.density_peaks_clustering(
            z,
            sigma_local=0.5,
            sigma_regional=1.0,
            min_bin_size=0.05,
            n_neighbors_search=self.dpc_split_kw.n_neighbors_search,
            remove_clusters_smaller_than=self.min_cluster_size,
            return_extra=False,
        )

        # prevent super oversplitting by checking centroid distances
        ids = np.unique(split_labels)
        ids = ids[ids >= 0]
        if ids.size <= 1:
            return 0

        new_units = []
        for label in ids:
            u = InterpUnit(do_interp=False, **self.unit_kw)
            inu = torch.tensor(in_unit[np.flatnonzero(split_labels == label)])
            inu, train_data = self.get_training_data(
                unit_id=None,
                in_unit=inu,
                sampling_method=self.sampling_method,
            )
            u.fit_center(**train_data, show_progress=False)
            new_units.append(u)
        kind = self.merge_metric
        min_overlap = self.min_overlap
        subset_channel_index = None
        if self.merge_on_waveform_radius:
            subset_channel_index = self.data.registered_reassign_channel_index
        nu = len(new_units)
        divergences = torch.full((nu, nu), torch.nan)
        for i, ua in enumerate(range(nu)):
            for j, ub in enumerate(range(nu)):
                if ua == ub:
                    divergences[i, j] = 0
                    continue
                divergences[i, j] = new_units[ua].divergence(
                    new_units[ub],
                    kind=kind,
                    min_overlap=min_overlap,
                    subset_channel_index=subset_channel_index,
                )
        dists = divergences.numpy(force=True)
        dists = np.maximum(dists, dists.T)

        dists[np.isinf(dists)] = dists[np.isfinite(dists)].max() + 10
        assert np.isfinite(dists).all(), f"{dists=}"
        d = dists[np.triu_indices(dists.shape[0], k=1)]
        Z = linkage(d, method=self.merge_linkage)
        new_labels = fcluster(Z, self.zip_threshold, criterion="distance")
        new_labels -= 1  # why do they do this...
        kept = split_labels >= 0
        split_labels[kept] = new_labels[split_labels[kept]]

        # -- deal with relabeling
        split_units, counts = np.unique(split_labels, return_counts=True)
        n_split_full = split_units.size
        counts = counts[split_units >= 0]
        assert (counts >= self.min_cluster_size // 2).all()
        split_units = split_units[split_units >= 0]
        n_split = split_units.size
        # case 0: nothing happened.
        if n_split <= 1:
            return 0  # , []

        # below, we want to re-fit this unit
        unit.needs_fit = True

        # case 2: something legitimately took place.
        # here, split_unit 0 retains label uid. split units >=1 get new labels.
        assert n_split in (n_split_full, n_split_full - 1)
        assert n_split > 1
        assert split_units[0] == 0
        self.labels[in_unit_full] = -1
        new_unit_ids = (
            unit_id,
            *(self.labels.max() + torch.arange(1, n_split, dtype=self.labels.dtype)),
        )
        for split_label, new_label in zip(split_units, new_unit_ids):
            self._zipper_parents[self.normalize_key(new_label)] = self.normalize_key(
                unit_id
            )
            in_split = in_unit[split_labels == split_label]
            self.labels[in_split] = new_label
        return len(new_unit_ids) - 1  # , new_unit_ids

    def continuity_split(self):
        if not self.do_interp:
            return

        n_new = 0
        for unit_id in tqdm(self.unit_ids(), desc="Continuity split"):
            n_new += self.continuity_split_unit(unit_id)
        print(f"Continuity split broke off {n_new} new units.")
        self.m_step()

    def continuity_split_unit(self, unit_id):
        if not self.do_interp:
            return {}

        unit = self[unit_id]
        if unit.interp.grid_fitted.sum() <= 1:
            return []

        times = unit.interp.grid.squeeze()
        times = times[unit.interp.grid_fitted]
        means = unit.get_means(times).reshape(len(times), -1)
        l2s = means.square().sum(1)

        if self.continuity_split_kwargs.scaled:
            dots = (means[:, None, :] * means[None, :, :]).sum(2)
            scalings = (dots + unit.inv_lambda).div_(l2s + unit.inv_lambda)
            scalings = scalings.clip_(unit.scale_clip_low, unit.scale_clip_high)
        else:
            scalings = torch.ones_like(l2s[:, None] + l2s[None, :])
        dists = (
            means[:, None]
            .sub(scalings[:, :, None] * means[None])
            .square()
            .sum(2)
            .div(l2s)
        )

        dists = dists.numpy(force=True)
        dists = np.maximum(dists, dists.T)
        d = dists[np.triu_indices(len(dists), k=1)]
        Z = linkage(d, method="complete")
        split_time_labels = fcluster(
            Z, self.continuity_split_kwargs.threshold, criterion="distance"
        )
        split_time_labels -= 1  # start at 0, not 1... what is this, Matlab?

        split_units = np.unique(split_time_labels)
        assert split_units[0] == 0
        n_split = split_units.size
        if n_split <= 1:
            return 0

        # assign spikes to nearest time points
        (in_unit,) = torch.nonzero(self.labels == unit_id, as_tuple=True)
        spike_times = self.data.times_seconds[in_unit]
        best_time_ix = (spike_times[:, None] - times[None]).abs().argmin(1)
        split_labels = split_time_labels[best_time_ix.numpy(force=True)]

        # check for size.
        split_units, split_counts = np.unique(split_labels, return_counts=True)
        big_enough = split_counts > self.min_cluster_size
        split_units = split_units[big_enough]
        n_split = split_units.size
        if n_split <= 1:
            return 0
        split_labels[np.logical_not(np.isin(split_labels, split_units))] = -1

        unit.needs_fit = True
        self.labels[in_unit] = -1
        new_unit_ids = (unit_id,)
        if n_split > 1:
            new_unit_ids = (
                *new_unit_ids,
                *(
                    self.labels.max()
                    + torch.arange(1, n_split, dtype=self.labels.dtype)
                ),
            )
        for split_label, new_label in zip(split_units, new_unit_ids):
            in_split = in_unit[split_labels == split_label]
            self.labels[in_split] = new_label
        return n_split - 1

    def unit_channel_overlaps(self, units_a=None, units_b=None):
        units = self.unit_ids()
        if units_a is None:
            units_a = units
        if units_b is None:
            units_b = units
        different = not torch.equal(units_a, units_b)
        masks_a = torch.zeros((len(units_a), len(self.data.registered_geom)))
        masks_b = masks_a
        if different:
            masks_b = torch.zeros((len(units_b), len(self.data.registered_geom)))
        for j, u in enumerate(units_a):
            masks_a[j, self[u].channels_valid] = 1.0
        if different:
            for j, u in enumerate(units_b):
                masks_b[j, self[u].channels_valid] = 1.0
        overlaps = masks_a @ masks_b.T
        overlaps /= masks_a.sum(1, keepdim=True)  # IoA
        return overlaps

    def central_divergences(
        self, units_a=None, units_b=None, kind=None, min_overlap=0.1, n_threads=0
    ):
        if kind is None:
            kind = self.merge_metric
        subset_channel_index = None
        if self.merge_on_waveform_radius:
            subset_channel_index = self.data.registered_reassign_channel_index
        if min_overlap is None:
            min_overlap = 1.0 - self.merge_threshold
        units = self.unit_ids()
        if units_a is None:
            units_a = units
        if units_b is None:
            units_b = units
        nua = len(units_a)
        nub = len(units_b)
        divergences = torch.full((nua, nub), torch.inf)
        divergences.diagonal().fill_(0.0)
        overlaps = self.unit_channel_overlaps(units_a, units_b).numpy(force=True)
        valid_pairs = [np.flatnonzero((o >= min_overlap) & (o > 0)) for o in overlaps]

        if not n_threads:
            for i, ua in enumerate(units_a):
                for j in valid_pairs[i]:
                    ub = units_b[j]
                    if ua == ub:
                        divergences[i, j] = 0
                        continue
                    divergences[i, j] = self[ua].divergence(
                        self[ub],
                        kind=kind,
                        min_overlap=min_overlap,
                        subset_channel_index=subset_channel_index,
                    )
            return divergences

        jobs = list(enumerate(units_a))

        def job(i, ua):
            ca = self[ua].channels_valid
            for j in valid_pairs[i]:
                ub = units_b[j]
                if ua == ub:
                    divergences[i, j] = 0
                    continue
                if torch.isin(self[ub].channels_valid, ca).sum() < (
                    self.merge_threshold * ca.numel()
                ):
                    continue
                divergences[i, j] = self[ua].divergence(
                    self[ub],
                    kind=kind,
                    min_overlap=min_overlap,
                    subset_channel_index=subset_channel_index,
                )

        results = joblib.Parallel(
            n_jobs=n_threads, backend="threading", return_as="generator"
        )(joblib.delayed(job)(*a) for a in jobs)
        results = tqdm(results, total=len(jobs), desc="pairwise")
        for _ in results:
            pass
        return divergences

    def unit_bimodalities(
        self,
        masked=True,
        weighted=True,
        mask_radius_s=5.0,
        n_threads=0,
        compute_mask=None,
    ):
        unit_ids = self.unit_ids()
        nu = len(unit_ids)

        scores = np.full((nu, nu), np.inf)
        np.fill_diagonal(scores, 0.0)
        memo = {}
        for i, ua in enumerate(unit_ids):
            (in_a,) = torch.nonzero(self.labels == ua, as_tuple=True)
            if masked:
                times_a = self.data.times_seconds[in_a][:, None]
                kdtree_a = KDTree(times_a.numpy(force=True))
                memo[self.normalize_key(ua)] = in_a.numpy(force=True), kdtree_a
            else:
                memo[self.normalize_key(ua)] = in_a.numpy(force=True), None

        for i, ua in enumerate(unit_ids):
            for j, ub in enumerate(unit_ids):
                if j <= i:
                    continue
                if compute_mask is not None:
                    if not compute_mask[i, j]:
                        continue

                in_a, kdtree_a = memo[self.normalize_key(ua)]
                in_b, kdtree_b = memo[self.normalize_key(ub)]

                if masked:
                    a_matched = kdtree_b.query(
                        kdtree_a.data, distance_upper_bound=mask_radius_s
                    )
                    a_matched = np.isfinite(a_matched[0])
                    b_matched = kdtree_a.query(
                        kdtree_b.data, distance_upper_bound=mask_radius_s
                    )
                    b_matched = np.isfinite(b_matched[0])
                    in_a = in_a[a_matched]
                    in_b = in_b[b_matched]

                na = in_a.size
                nb = in_b.size
                if min(na, nb) < 2:
                    continue
                ntot = na + nb

                which = np.concatenate((in_a, in_b))
                order = np.argsort(which)
                which = which[order]
                if weighted:
                    sample_weights = np.zeros(which.shape)
                    sample_weights[:na] = (nb / ntot) / 0.5
                    sample_weights[na:] = (na / ntot) / 0.5
                    sample_weights = sample_weights[order]
                else:
                    sample_weights = np.ones(which.shape)

                # reassignment scores between these two units
                badness = self.reassignment_divergences(
                    which_spikes=torch.from_numpy(which).to(self.labels),
                    unit_ids=[ua, ub],
                    show_progress=False,
                )
                # replace sparse zeros with 1s (explicit 0s are cool though)
                tmp_dense = np.full(badness.shape, np.inf)
                tmp_dense[badness.coords] = badness.data
                tmp_dense = np.nan_to_num(tmp_dense, nan=1.0, posinf=1.0, copy=False)
                bad_a, bad_b = tmp_dense
                dbad = bad_a - bad_b
                unique_dbad, inverse = np.unique(dbad, return_inverse=True)

                weights = np.zeros(unique_dbad.shape)
                np.add.at(weights, inverse, sample_weights)
                score_ab = smoothed_dipscore_at(
                    0.0, unique_dbad, weights, dipscore_only=True
                )
                scores[i, j] = scores[j, i] = score_ab

        return scores

    def merge(self, n_threads=0):
        merge_dists = self.central_divergences(
            kind=self.merge_metric,
            # min_overlap=self.min_overlap,
            n_threads=n_threads,
        )
        merge_dists = self.merge_sym_function(merge_dists, merge_dists.T)
        merge_dists = merge_dists.numpy(force=True)

        if self.merge_with_bimodality:
            to_check = np.isfinite(merge_dists)
            bimodalities = self.unit_bimodalities(compute_mask=to_check)
            # change bimodality so that thresholds match
            bimodalities *= self.merge_threshold / self.bimodality_threshold
            merge_dists = np.maximum(bimodalities, merge_dists)

        valid = np.isfinite(merge_dists)
        merge_dists[np.logical_not(valid)] = merge_dists[valid].max() + 10
        d = merge_dists[np.triu_indices(merge_dists.shape[0], k=1)]
        Z = linkage(d, method=self.merge_linkage)
        new_labels = fcluster(Z, self.merge_threshold, criterion="distance")
        new_labels -= 1
        unique_new_labels = np.unique(new_labels)
        print(f"Merge: {merge_dists.shape[0]} -> {unique_new_labels.size}")

        # update state
        self.update_labels(
            self.unit_ids(),
            torch.from_numpy(new_labels).to(self.labels),
        )
        self.order_by_depth()

    def all_reassignment_divergences(
        self,
        show_progress=True,
        kind=None,
        n_threads=1,
        exclude_above=None,
    ):
        if kind is None:
            kind = self.reassign_metric

        inds_units = [(j, self[uid]) for j, uid in enumerate(self.unit_ids())]
        pool = joblib.Parallel(n_threads, backend="threading", return_as="generator")
        n_batches = int(np.ceil(self.data.n_spikes / self.batch_size))

        @joblib.delayed
        def job(j, unit, sl, times, waveforms, waveform_channels):
            overlaps, rel_ix = unit.overlaps(waveform_channels)
            (which,) = torch.nonzero(overlaps >= self.min_overlap, as_tuple=True)
            if not which.numel():
                return None, None, None

            overlaps = overlaps[which]
            rel_ix = rel_ix[which]
            _, _, badnesses = unit.spike_badnesses(
                times=times[which],
                waveforms=waveforms[which],
                waveform_channels=waveform_channels[which],
                overlaps=overlaps,
                rel_ix=rel_ix,
                kinds=(kind,),
            )
            badnesses = badnesses[kind]

            keep = slice(None)
            if exclude_above is not None:
                (keep,) = torch.nonzero(badnesses <= exclude_above, as_tuple=True)
                if not keep.size:
                    return None, None, None

            return j, sl.start + which[keep].numpy(force=True), badnesses[keep].numpy(force=True)

        ii = []
        jj = []
        values = []
        import itertools
        jobs = self.all_batches(waveform_kind="reassign")
        jobs = itertools.product(jobs, inds_units)
        # for sl, data in tqdm(self.all_batches(waveform_kind="reassign"), desc="Reassign batches", total=n_batches):
        #     for j, which, results in pool(job(*ju, sl, **data) for ju in inds_units):
        jobs = pool(
            job(*ju, sl, **data) for (sl, data), ju in jobs
        )
        jobs = tqdm(jobs, desc="Reassign", total=n_batches * len(inds_units))
        for j, which, results in jobs:
            ii.append(np.broadcast_to(np.array([j]), which.shape))
            jj.append(which)
            values.append(results)

        me = threading.get_ident()
        need_alloc = me not in self._reas_bufs
        nout = sum(v.size for v in values)
        if not need_alloc:
            if nout > self._reas_bufs[me][0].size:
                need_alloc = True
                del self._reas_bufs[me]
        if need_alloc:
            nalloc = int(np.ceil(nout * 1.25))
            vout = np.empty(nalloc, dtype=dtype)
            iiout = np.empty(nalloc, dtype=np.int32)
            jjout = np.empty(nalloc, dtype=np.int32)
            self._reas_bufs[me] = vout, iiout, jjout
        vout, iiout, jjout = self._reas_bufs[me]
        if nout:
            values = np.concatenate(values, out=vout[:nout])
            ii = np.concatenate(ii, out=iiout[:nout])
            jj = np.concatenate(jj, out=jjout[:nout])
            divergences = coo_array((values, (ii, jj)), dtype=dtype, shape=shape)
        else:
            divergences = coo_array(shape, dtype=dtype)

        return divergences


    def reassignment_divergences(
        self,
        which_spikes=None,
        unit_ids=None,
        units=None,
        show_progress=True,
        n_threads=0,
        exclude_above=None,
        kind=None,
        single=False,
    ):
        if unit_ids is None and units is None:
            unit_ids = self.unit_ids()
        if units is None:
            units = [self[uid] for uid in unit_ids]
        if kind is None:
            kind = self.reassign_metric
        nu = len(units)

        subsampling = which_spikes is not None
        if subsampling:
            n_spikes = which_spikes.numel()
        else:
            n_spikes = self.data.n_spikes
            which_spikes = slice(None)

        dtype = self.data.original_tpca_embeds.dtype
        dtype = str(dtype).split(".")[-1]
        shape = (nu, n_spikes)
        ii = []
        jj = []
        values = []

        if show_progress:
            units = tqdm(units, desc="Spike divergences", **tqdm_kw)

        static_chans = self.data.reassign_static_channels[which_spikes]
        if n_threads == 0:
            for j, unit in enumerate(units):
                overlaps, rel_ix = unit.overlaps(static_chans)
                (which,) = torch.nonzero(overlaps >= self.min_overlap, as_tuple=True)
                if not which.numel():
                    continue
                overlaps = overlaps[which]
                rel_ix = rel_ix[which]
                batch_indices = which = which.numpy(force=True)
                if subsampling:
                    batch_indices = which_spikes[which]

                results = np.zeros(which.shape, dtype=dtype)
                for sl, batch in self.batches(batch_indices, waveform_kind="reassign"):
                    _, _, res = unit.spike_badnesses(
                        **batch,
                        overlaps=overlaps[sl],
                        rel_ix=rel_ix[sl],
                        kinds=(kind,),
                    )
                    results[sl] = res[kind].numpy(force=True)
                    # divergences[j, which[sl]] = res[kind].numpy(force=True)
                keep = slice(None)
                if exclude_above is not None:
                    keep = np.flatnonzero(results <= exclude_above)
                    if not keep.size:
                        continue
                jj.append(which[keep])
                ii.append(np.broadcast_to(np.array([j]), jj[-1].shape))
                values.append(results[keep])
        else:
            import joblib

            def reas_job(j, unit):
                # j, unit = j__unit
                overlaps, rel_ix = unit.overlaps(static_chans)
                (which,) = torch.nonzero(overlaps >= self.min_overlap, as_tuple=True)
                if not which.numel():
                    return None, None, None
                overlaps = overlaps[which]
                rel_ix = rel_ix[which]
                batch_indices = which = which.numpy(force=True)
                if subsampling:
                    batch_indices = which_spikes[which]

                results = np.zeros(which.shape, dtype=dtype)
                rrr = []
                for sl, batch in self.batches(batch_indices, waveform_kind="reassign"):
                    _, _, res = unit.spike_badnesses(
                        **batch,
                        overlaps=overlaps[sl],
                        rel_ix=rel_ix[sl],
                        kinds=(self.reassign_metric,),
                    )
                    rrr.append((sl, res[self.reassign_metric]))
                for sl, res in rrr:
                    results[sl] = res.numpy(force=True)

                keep = slice(None)
                if exclude_above is not None:
                    keep = np.flatnonzero(results <= exclude_above)
                    if not keep.size:
                        return None, None, None
                return j, which[keep], results[keep]

            for j, which, results in joblib.Parallel(
                n_jobs=n_threads,
                backend="threading",
                return_as="generator",
            )(joblib.delayed(reas_job)(jj, uu) for jj, uu in enumerate(units)):
                if j is None:
                    continue
                # divergences[j, which] = results
                ii.append(np.broadcast_to(np.array([j]), which.shape))
                jj.append(which)
                values.append(results)

        me = threading.get_ident()
        need_alloc = me not in self._reas_bufs
        nout = sum(v.size for v in values)
        if not need_alloc:
            if nout > self._reas_bufs[me][0].size:
                need_alloc = True
                del self._reas_bufs[me]
        if need_alloc:
            nalloc = int(np.ceil(nout * 1.25))
            vout = np.empty(nalloc, dtype=dtype)
            iiout = np.empty(nalloc, dtype=np.int32)
            jjout = np.empty(nalloc, dtype=np.int32)
            self._reas_bufs[me] = vout, iiout, jjout
        vout, iiout, jjout = self._reas_bufs[me]
        if nout:
            values = np.concatenate(values, out=vout[:nout])
            ii = np.concatenate(ii, out=iiout[:nout])
            jj = np.concatenate(jj, out=jjout[:nout])
            divergences = coo_array((values, (ii, jj)), dtype=dtype, shape=shape)
        else:
            divergences = coo_array(shape, dtype=dtype)

        return divergences

    def reassignment_weights(
        self,
        which_spikes=None,
        unit_ids=None,
        units=None,
        show_progress=True,
        n_threads=0,
        exclude_above=None,
        kind=None,
        single=False,
        divergences=None,
        max_weight=1e2,
    ):
        div_kind = kind
        if kind == "recipr2":
            div_kind = "1-r^2"
            kind = "recip"
        if divergences is None:
            divergences = self.reassignment_divergences(
                which_spikes=which_spikes,
                unit_ids=unit_ids,
                units=units,
                show_progress=show_progress,
                n_threads=n_threads,
                exclude_above=exclude_above,
                kind=div_kind,
                single=single,
            )

        if "recip" in kind:
            divergences.data.clip(min=1.0 / max_weight, out=divergences.data)
            np.reciprocal(divergences.data, out=divergences.data)

        # convert to torch sparse so we can do a softmax
        coo = torch.from_numpy(divergences.coords[0]), torch.from_numpy(
            divergences.coords[1]
        )
        coo = torch.row_stack(coo)
        weights = torch.sparse_coo_tensor(
            coo,
            torch.from_numpy(divergences.data).to(torch.float),
            size=divergences.shape,
        )
        weights = weights.to(self.device)

        if kind in ("l2", "scaledl2"):
            weights = torch.sparse.softmax(-0.5 * weights, dim=0)
        elif kind == "1-r^2":
            weights = torch.sparse.softmax(-5 * weights, dim=0)
        elif kind == "recip":
            pass
        else:
            assert False
        return weights

    def reassign(
        self, n_threads=0, show_progress=True, verbose=True, return_divergences=False, drop_dups=False
    ):
        divergences = self.reassignment_divergences(
            n_threads=n_threads,
            exclude_above=self.match_threshold,
            show_progress=show_progress,
        )
        divergences_csc, new_labels = sparse_reassign(divergences, None, return_csc=True)

        if drop_dups:
            print("drop dups!")
            for unit_id in np.unique(new_labels):
                in_unit = np.flatnonzero(new_labels == unit_id)
                my_times = self.data.times_samples[in_unit]
                unique_times, counts = np.unique(my_times, return_counts=True)
                viol = np.flatnonzero(counts > 0)
                if not viol.size:
                    continue
                for j in viol:
                    bad_ix = in_unit[my_times == unique_times[j]]
                    divs = divergences_csc[np.full_like(bad_ix, unit_id), bad_ix]
                    best_of_the_bad = divs.min()
                    new_labels[bad_ix[divs > best_of_the_bad]] = -1

        outlier_pct = 100 * (new_labels < 0).mean()
        if verbose:
            print(f"Reassignment marked {outlier_pct:.1f}% of spikes as outliers.")

        new_labels = torch.as_tensor(
            new_labels, dtype=self.labels.dtype, device=self.labels.device
        )
        reas_pct = 100 * (self.labels != new_labels).to(torch.float).mean().numpy(
            force=True
        )
        if verbose:
            print(f"{reas_pct:.1f}% of spikes reassigned")
        self.labels.copy_(new_labels)
        keep_mask = self.cleanup()

        if not return_divergences:
            return outlier_pct, reas_pct, None

        # return responsibilities for caller
        # thing is, we need to re-index the rows after the cleanup
        kept_label_indices = np.flatnonzero(keep_mask.numpy(force=True))
        if kept_label_indices.size < divergences.shape[0]:
            ii, jj = divergences.coords
            ixs = np.searchsorted(kept_label_indices, ii)
            ixs.clip(0, kept_label_indices.size - 1, out=ixs)
            valid = np.flatnonzero(kept_label_indices[ixs] == ii)
            divergences = coo_array(
                (divergences.data[valid], (ixs[valid], jj[valid])),
                shape=(kept_label_indices.size, divergences.shape[1]),
            )

        return outlier_pct, reas_pct, divergences

    def get_indices(self, uid, n=None, in_unit=None, sampling_method=None):
        if n is None:
            n = self.n_spikes_fit
        if sampling_method is None:
            sampling_method = self.sampling_method

        if in_unit is None:
            (in_unit,) = (self.labels == uid).nonzero(as_tuple=True)
        ns = in_unit.numel()
        if ns <= n:
            return in_unit

        if sampling_method == "random":
            which = self.rg.choice(ns, size=n, replace=False)
            which.sort()
        elif sampling_method == "time_amp_reweighted":
            # density ratio
            times = self.data.times_seconds[in_unit].numpy(force=True)
            amps = self.data.amps[in_unit]
            x = np.c_[times / mad(times), amps / mad(amps)]
            dens = density.get_smoothed_densities(x, sigmas=self.sampling_sigma)
            p = np.reciprocal(dens, out=dens)
            p /= p.sum()
            which = self.rg.choice(ns, size=n, p=p, replace=True)
            which.sort()
        else:
            assert False

        in_unit = in_unit[torch.from_numpy(which)]
        return in_unit

    def get_training_data(
        self,
        unit_id,
        n=None,
        in_unit=None,
        waveform_kind="original",
        sampling_method=None,
    ):
        in_unit = self.get_indices(unit_id, n=n, in_unit=in_unit)
        train_data = self.spike_data(in_unit, waveform_kind=waveform_kind)
        train_data["geom"] = self.data.registered_geom
        if self.channel_strategy != "snr":
            train_data["cluster_channel_index"] = self.data.cluster_channel_index
        if waveform_kind == "original":
            train_data["waveform_channel_index"] = (
                self.data.registered_original_channel_index
            )
        elif waveform_kind == "reassign":
            train_data["waveform_channel_index"] = (
                self.data.registered_reassign_channel_index
            )
        else:
            assert False
        return in_unit, train_data

    def spike_data(self, which, waveform_kind="original"):
        if waveform_kind == "original":
            ssc = self.data.original_static_channels[which]
        elif waveform_kind == "reassign":
            ssc = self.data.reassign_static_channels[which]
        else:
            assert False
        waveforms = self.data.get_waveforms(
            which, device=self.device, kind=waveform_kind
        )
        ssc = ssc.to(self.device)
        return dict(
            times=self.data.times_seconds[which],
            waveforms=waveforms,
            waveform_channels=ssc,
        )

    def all_batches(self, waveform_kind="original", batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        for j in range(0, self.data.n_spikes, batch_size):
            sl = slice(j, min(j + batch_size, self.data.n_spikes))
            yield sl, self.spike_data(sl, waveform_kind=waveform_kind)

    def batches(self, indices, batch_size=None, waveform_kind="original"):
        if batch_size is None:
            batch_size = self.batch_size
        for j in range(0, len(indices), batch_size):
            sl = slice(j, min(j + batch_size, len(indices)))
            yield sl, self.spike_data(indices[sl], waveform_kind=waveform_kind)


# -- core classes


class MaskedPCA(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        rank=2,
        max_iter=100,
        check_every=5,
        n_oversamples=10,
        atol=1e-3,
        centered=True,
        transform_iter=0,
    ):
        super().__init__()
        self.fit_kw = dict(
            max_iter=max_iter,
            check_every=check_every,
            n_oversamples=n_oversamples,
            atol=atol,
            centered=centered,
        )
        self.rank = rank
        self.transform_iter = transform_iter
        self.train_loadings = None
        self.centered = centered
        if centered:
            self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("weight", torch.zeros((rank, input_dim)))
        self.register_buffer("svs", torch.zeros((rank)))

    def fit(self, waveforms, show_progress=False):
        if not waveforms.numel():
            return
        missing = torch.isnan(waveforms)
        empty = missing.all(1)
        loadings, mean, components, svs = fit_pcas(
            waveforms,
            missing,
            empty,
            rank=self.rank,
            show_progress=show_progress,
            **self.fit_kw,
        )
        self.train_loadings = loadings
        self.weight.copy_(components.T)
        if self.centered:
            self.mean.copy_(mean)
        self.svs.copy_(svs)

    def forward_precentered(self, waveforms, out=None):
        return torch.matmul(waveforms, self.weight.T, out=out)

    def forward(self, waveforms):
        if self.centered:
            waveforms = waveforms - self.mean
        return waveforms @ self.weight.T

    def backward_precentered(self, embeds):
        return embeds @ self.weight

    def backward(self, embeds):
        if self.centered:
            return torch.addmm(self.mean, embeds, self.weight)
        return embeds @ self.weight

    def transform(self, waveforms):
        mask = torch.isfinite(waveforms)
        waveforms = torch.where(
            mask,
            waveforms,
            self.mean,
        )

        if self.transform_iter == 0:
            return self.forward(waveforms)

        for j in range(self.transform_iter):
            embeds = self.forward(waveforms)
            recons = self.backward(embeds)
            waveforms = torch.where(
                mask,
                waveforms,
                recons,
                out=waveforms,
            )

        return embeds

    def transform_precentered(self, waveforms):
        mask = torch.isfinite(waveforms)
        waveforms = torch.nan_to_num(waveforms)

        if self.transform_iter == 0:
            return self.forward_precentered(waveforms)

        for j in range(self.transform_iter):
            embeds = self.forward_precentered(waveforms)
            recons = self.backward_precentered(embeds)
            waveforms = torch.where(
                mask,
                waveforms,
                recons,
                out=waveforms,
            )

        return embeds


class InterpFactorAnalysis(torch.nn.Module):
    def __init__(
        self,
        t_bounds,
        output_dim,
        latent_dim=1,
        lengthscale=100.0,
        prior_noiselogit=-6.0,
        obs_logstd=0.0,
        points_per_lengthscale=2,
        grid_size=None,
        learn_lengthscale=False,
        min_lengthscale=1e-5,
        learn_prior_noise_fraction=False,
        learn_obsstd=True,
        loss_on_interp=False,
        latent_update="gradient",
        do_prior=True,
        fitted_point_count=25,
        fitted_point_fraction="grid",
        interp_kind="cubic",
    ):
        super().__init__()
        self.t_bounds = t_bounds
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.loss_on_interp = loss_on_interp
        self.interp_kind = interp_kind

        # make grid
        if grid_size is None:
            grid_size = (
                (t_bounds[1] - t_bounds[0]) / lengthscale
            ) * points_per_lengthscale
            grid_size = int(np.ceil(grid_size))
        self.grid_size = grid_size
        self.register_buffer(
            "grid",
            torch.linspace(
                torch.tensor(t_bounds[0], dtype=torch.float),
                torch.tensor(t_bounds[1], dtype=torch.float),
                steps=grid_size,
            )[:, None],
        )
        grid_z = torch.zeros((grid_size, latent_dim))
        self.latent_update = latent_update
        if latent_update == "gradient":
            self.register_parameter("grid_z", torch.nn.Parameter(grid_z))
        elif latent_update == "embed_uninterp":
            self.register_buffer("grid_z", grid_z)
        else:
            assert False

        self.fitted_point_count = fitted_point_count
        self.fitted_point_fraction = fitted_point_fraction
        if fitted_point_fraction == "grid":
            self.fitted_point_fraction = 1 / self.grid_size

        self.do_prior = do_prior
        self.learn_lengthscale = learn_lengthscale
        self.learn_prior_noise_fraction = learn_prior_noise_fraction
        self.compute_prior_in_loss = False
        self.init_kernel(lengthscale, min_lengthscale)
        if self.do_prior:
            self.register_buffer("prior_mean", torch.zeros(()))
            prior_noiselogit = prior_noiselogit + torch.zeros(())
            if learn_prior_noise_fraction:
                self.register_parameter(
                    "prior_noiselogit", torch.nn.Parameter(prior_noiselogit)
                )
            else:
                self.register_buffer("prior_noiselogit", prior_noiselogit)
            self._interp_id = None
            self._cached_priordist = None
            if self.learn_lengthscale or self.learn_prior_noise_fraction:
                self.compute_prior_in_loss = True
            if self.latent_update == "gradient":
                self.compute_prior_in_loss = True

        # observation model
        self.net = torch.nn.Linear(latent_dim, self.output_dim)
        obs_logstd = obs_logstd + torch.zeros(output_dim)
        if learn_obsstd:
            self.register_parameter(
                "obs_logstd",
                torch.nn.Parameter(obs_logstd),
            )
        else:
            self.register_buffer(
                "obs_logstd",
                obs_logstd,
            )
        self._unbias = None
        self._unweight = None

    def init_kernel(self, lengthscale, min_lengthscale):
        assert lengthscale >= min_lengthscale
        if not self.learn_lengthscale:
            Kuu = RBF(lengthscale)(self.grid)
            self.register_buffer("_grid_cov", torch.tensor(Kuu, dtype=torch.float))
            if self.do_prior:
                grid_scale_left = np.linalg.cholesky(Kuu)
                self.register_buffer(
                    "_grid_scale_left", torch.tensor(grid_scale_left, dtype=torch.float)
                )
        else:
            self.register_buffer(
                "_half_sq_dgrid", -0.5 * torch.square(self.grid - self.grid.T)
            )

        lengthscale = torch.tensor(lengthscale)
        min_lengthscale = torch.tensor(min_lengthscale)
        self.register_buffer("min_lengthscale", min_lengthscale)
        if self.learn_lengthscale:
            lengthscale = lengthscale - self.min_lengthscale
            if lengthscale < 20.0:
                # invert softplus
                lengthscale = lengthscale.expm1().log()
            self.register_parameter("_lengthscale", torch.nn.Parameter(lengthscale))
        else:
            self.register_buffer("_lengthscale", lengthscale)

    def lengthscale(self):
        if self.learn_lengthscale:
            return F.softplus(self._lengthscale) + self.min_lengthscale
        return self._lengthscale

    def grid_cov(self):
        if not self.learn_lengthscale:
            return self._grid_cov

        Kuu = self._half_sq_dgrid / self.lengthscale().square()
        Kuu = Kuu.exp()
        return Kuu

    def grid_scale_left(self, eps=1e-4):
        if not self.learn_lengthscale:
            return self._grid_scale_left
        Kuu = self.grid_cov()
        # Kuu.diagonal().add_(eps)
        # Kuu = Kuu + eps * torch.eye(self.grid_size, device=Kuu.device)
        scale_left = torch.linalg.cholesky(Kuu)
        return scale_left

    def noise_fraction(self):
        return F.sigmoid(self.prior_noiselogit)

    @torch.no_grad()
    def _compute_grid_matrix(self, inputs):
        left_interp_matrix = left_interpolation_matrix(
            self.grid, inputs, kind=self.interp_kind
        )
        return left_interp_matrix

    def forward(self, t, left_interp_matrix=None):
        if left_interp_matrix is None:
            left_interp_matrix = self._compute_grid_matrix(t)
        z = left_interp_matrix @ self.grid_z
        preds = self.net(z)
        return z, preds

    def get_prior_distribution(self, left_interp_matrix=None, eps=1e-4):
        if not (self.learn_lengthscale or self.learn_prior_noise_fraction):
            if self._cached_priordist is not None:
                if self._interp_id == id(left_interp_matrix):
                    return self._cached_priordist
                else:
                    self._cached_priordist = None

        scale_left = self.grid_scale_left(eps)
        if self.loss_on_interp:
            scale_left = left_interp_matrix @ scale_left

        n_prior = left_interp_matrix.shape[0] if self.loss_on_interp else self.grid_size

        lambd = self.noise_fraction()
        prior_dist = torch.distributions.LowRankMultivariateNormal(
            self.prior_mean.broadcast_to((n_prior,)),
            (1.0 - lambd) * scale_left,
            lambd.add(eps).broadcast_to((n_prior,)),
        )
        if not (self.learn_lengthscale or self.learn_prior_noise_fraction):
            self._cached_priordist = prior_dist
            self._interp_id = id(left_interp_matrix)
        return prior_dist

    def log_prior(self, z, left_interp_matrix=None, eps=1e-4):
        prior_dist = self.get_prior_distribution(left_interp_matrix, eps)
        if self.loss_on_interp:
            logprior = prior_dist.log_prob(z.T).sum()
        else:
            logprior = prior_dist.log_prob(self.grid_z.T).sum()
        return logprior, prior_dist

    @torch.no_grad()
    def update_z_embed_uninterp(
        self,
        y,
        prior_dist=None,
        left_interp_matrix=None,
        left_interp_pinv=None,
        mask=None,
    ):
        assert self.training

        if self.do_prior and prior_dist is None:
            prior_dist = self.get_prior_distribution(left_interp_matrix)
        if left_interp_pinv is None:
            left_interp_pinv = torch.linalg.pinv(left_interp_matrix)
        z = self.embed(y, mask=mask)
        z = left_interp_pinv @ z
        if self.do_prior:
            assert False
            # z = torch.linalg.solve_triangular(prior_dist._capacitance_tril, z, upper=False)
            # z = torch.cholesky_solve(z, prior_dist._capacitance_tril, )
            # z = left_interp_matrix @ z
            # z = left_interp_pinv @ z
        self.grid_z.copy_(z)

    def embed(self, y, mask=None):
        if self.training:
            try:
                unweight = torch.linalg.pinv(self.net.weight).T
            except Exception as e:
                print("bad")
                print(f"{torch.isfinite(self.net.bias).all()=}")
                print(f"{torch.isfinite(self.net.weight).all()=}")
                print(
                    f"{torch.isfinite(torch.where(mask, y, self.net.bias[None])).all()=}"
                )
                print(f"{y.shape=}")
                raise
        else:
            if self._unweight is None:
                self._unweight = torch.linalg.pinv(self.net.weight).T
            unweight = self._unweight
        if mask is not None:
            y = torch.where(mask, y, self.net.bias[None])
        # print(f"{torch.(self.net.bias).all()=}")
        # print(f"{torch.isfinite(self.net.weight).all()=}")
        # print(f"{torch.isisfinitefinite(y).all()=}")
        # print(f"{y.shape=}")
        return (y - self.net.bias) @ unweight

    def log_likelihood(self, preds, targets, mask_tuple):
        obs_var = (2 * self.obs_logstd).exp()
        recon_err = torch.square(preds[mask_tuple] - targets[mask_tuple])
        recon_err = (recon_err / obs_var[mask_tuple[1]]).sum()
        denom = (self.obs_logstd[mask_tuple[1]]).sum()
        loglik = -0.5 * recon_err - denom

        return loglik

    def loss(self, train_t, train_y, left_interp_matrix, mask_tuple, eps=1e-4):
        z, pred = self(train_t, left_interp_matrix)
        loss = -self.log_likelihood(pred, train_y, mask_tuple)

        prior_dist = None
        if self.compute_prior_in_loss:
            logprior, prior_dist = self.log_prior(z, left_interp_matrix, eps)
            loss = loss - logprior

        return loss, prior_dist

    def initialize_svd_smoothed(
        self, train_t, train_y, left_interp_matrix, missing=None, empty=None
    ):
        if missing is None:
            missing = torch.isnan(train_y)
        if empty is None:
            empty = missing.all(1)
        loadings, mean, components, svs = fit_pcas(
            train_y, missing, empty, self.latent_dim, max_iter=1
        )
        grid_cov = self.grid_cov()
        weights = left_interp_matrix @ grid_cov
        weights = torch.nan_to_num(weights / weights.sum(0))
        zinit = weights.T @ loadings

        if not torch.isfinite(zinit).all():
            print(f"{weights0.sum()=}")
            print(f"{weights0.min()=}")
            print(f"{weights0.max()=}")
            print(f"{weights0.sum(0)=}")
            print(f"{weights0.sum(0).min()=}")
            print(f"{weights0.sum(0).max()=}")
            print(f"{self.grid.min()=}")
            print(f"{self.grid.max()=}")
            print(f"{train_t.min()=}")
            print(f"{train_t.max()=}")

        with torch.no_grad():
            self.net.bias.copy_(mean)
            self.net.weight.copy_(components)
            self.grid_z.copy_(zinit)

    def fit(
        self,
        train_t,
        train_y,
        lr=0.05,
        eps=1e-6,
        show_progress=True,
        n_iter=200,
        loss_converged=1e-2,
    ):
        assert self.training

        # precompute cubic interpolation kernel info
        left_interp_matrix = self._compute_grid_matrix(train_t)
        if self.latent_update == "embed_uninterp":
            left_interp_pinv = torch.linalg.pinv(left_interp_matrix)

        # missing values info
        mask = torch.isfinite(train_y)
        mask_tuple = torch.nonzero(mask, as_tuple=True)
        missing = torch.logical_not(mask)
        empty = missing.all(1)

        # initialize with pca
        self.initialize_svd_smoothed(
            train_t, train_y, left_interp_matrix, missing, empty
        )

        # optimize
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        losses = torch.zeros(n_iter, device=self.net.bias.device, requires_grad=False)
        for i in (trange if show_progress else range)(n_iter):
            opt.zero_grad()
            try:
                loss, prior_dist = self.loss(
                    train_t, train_y, left_interp_matrix, mask_tuple, eps
                )
            except:
                print(f"Exc at {i=}")
                print(f"{self.grid=}")
                print(f"{self.grid_z=}")
                raise
            losses[i] = loss.detach()
            if i > 10 and loss + loss_converged >= losses[i - 1]:
                break

            loss.backward()
            opt.step()

            if self.latent_update == "embed_uninterp":
                self.update_z_embed_uninterp(
                    train_y,
                    prior_dist=prior_dist,
                    left_interp_matrix=left_interp_matrix,
                    left_interp_pinv=left_interp_pinv,
                    mask=mask,
                )

        # check which grid points had enough spikes
        grid_neighbors = torch.cdist(
            train_t[:, None],
            self.grid,
            p=1,
        ).argmin(dim=1)
        histogram = torch.zeros(
            self.grid_size, device=grid_neighbors.device, dtype=grid_neighbors.dtype
        )
        histogram.scatter_add_(
            0,
            grid_neighbors,
            torch.ones(
                1, dtype=grid_neighbors.dtype, device=grid_neighbors.device
            ).broadcast_to(grid_neighbors.shape),
        )
        self.register_buffer(
            "grid_fitted",
            torch.logical_or(
                histogram >= self.fitted_point_count,
                histogram / histogram.sum() >= self.fitted_point_fraction,
            ),
        )

        return losses[: i + 1].numpy(force=True)


# -- core functions


@torch.no_grad()
def fit_pcas(
    X,
    missing,
    empty,
    rank,
    max_iter=100,
    check_every=5,
    n_oversamples=10,
    atol=1e-3,
    show_progress=False,
    centered=True,
):
    """
    X : (nu, n, dim_in)
    missing : (nu, n, dim_in)
    empty : (nu, n)
    """
    # single = X.ndim == 2
    # if single:
    #     assert missing.ndim == 2
    #     assert empty.ndim == 1
    #     X = X[None]
    #     missing = missing[None]
    #     empty = empty[None]

    # initialize mean
    # Xc = torch.where(ignore, torch.nan, X)
    Xc = X.clone()
    if centered:
        mean = torch.nan_to_num(Xc.nanmean(dim=-2, keepdims=True))
    else:
        shape = list(Xc.shape)
        shape[-1] = 1
        mean = torch.zeros(shape, device=Xc.device, dtype=Xc.dtype)
    # after this line, isnan(Xc) === empty.
    if missing is None:
        missing = torch.isnan(X)
        empty = missing.all(1)
    Xc[missing] = mean.broadcast_to(X.shape)[missing]
    if centered:
        mean = Xc.nanmean(dim=-2, keepdims=True)
    else:
        Xc = torch.where(empty[..., None], 0, Xc)

    ###
    filled = torch.logical_not(empty)
    no_missing = not missing[filled].any()
    addmm = torch.baddbmm if X.ndim == 3 else torch.addmm

    # iterate svds
    it = trange(max_iter, desc="SVD") if show_progress else range(max_iter)
    # svd_storage = None
    q = min(rank + n_oversamples, X.shape[-1])
    for j in it:
        # update svd
        if centered:
            Xin = torch.where(empty[..., None], 0, Xc - mean)
        else:
            Xin = Xc
        # svd_storage = torch.linalg.svd(Xin, full_matrices=False, out=svd_storage)
        # U, S, Vh = svd_storagev

        # faster in my experience despite the note in torch.linalg.svd docs
        U, S, V = torch.svd_lowrank(Xin, q=q)
        Vh = V.mT

        U = U[..., :rank]
        S = S[..., :rank]
        Vh = Vh[..., :rank, :]

        if no_missing:
            break

        # impute
        recon = addmm(mean, U, S[..., None] * Vh)
        check = not (j % check_every)
        if check:
            dx = (Xc[missing] - recon[missing]).abs().max().numpy(force=True)
            dx = float(dx)
            if show_progress:
                it.set_description(f"{dx=:0.5f}")
        Xc[missing] = recon[missing]
        if centered:
            mean = Xc.nanmean(dim=-2, keepdims=True)

        if check and dx < atol:
            break

    # svd -> pca
    loadings = U * S[..., None, :]
    mean = mean[..., 0, :]
    components = Vh.mT
    svs = S

    # if single:
    #     loadings = loadings[0]
    #     mean = mean[0]
    #     components = components[0]
    #     svs = svs[0]

    return loadings, mean, components, svs


def _load_data(
    sorting,
    motion_est,
    fit_radius,
    reassign_wf_radius=None,
    in_memory=False,
    keep="all",
    max_n_spikes=5000000,
    load_tpca=False,
    on_device=False,
    pin=False,
    tpca_feature_name="collisioncleaned_tpca_features",
    interp_kind="nearest",
    drift_positions="channel",
    kriging_sigma="pitch",
    rg=0,
):
    rg = np.random.default_rng(rg)

    # load up labels
    labels = sorting.labels
    amps = sorting.denoised_ptp_amplitudes
    if keep == "labeled":
        keep_mask = labels >= 0
    elif keep == "all":
        keep_mask = np.ones(labels.shape, dtype=bool)
    elif keep == "byamp":
        keep_mask = labels >= 0
        keep_mask = np.logical_or(
            keep_mask,
            amps >= np.median(amps[keep_mask]),
        )
    keepers = np.flatnonzero(keep_mask)

    if max_n_spikes and keepers.size > max_n_spikes:
        print(
            f"Subsampling from {keepers.size} to {max_n_spikes} ({100*(max_n_spikes/labels.size):0.1f}%)"
        )
        keepers = rg.choice(keepers, size=max_n_spikes, replace=False)
        keepers.sort()
        keep_mask = np.zeros_like(keep_mask)
        keep_mask[keepers] = 1

    labels = labels[keepers]
    channels = sorting.channels[keepers]
    times_seconds = sorting.times_seconds[keepers]
    amps = amps[keepers]
    times_samples = sorting.times_samples[keepers]

    # load waveforms and subset by radius, retaining the new index for later
    h5 = h5py.File(sorting.parent_h5_path, "r", locking=False)
    geom = h5["geom"][:]
    original_channel_index = h5["channel_index"][:]
    original_radius = 0
    for i in range(len(geom)):
        cii = original_channel_index[i]
        i_rad = np.square(geom[i] - geom[cii[cii < len(geom)]]).sum(1)
        original_radius = max(np.sqrt(i_rad.max()), original_radius)
    if drift_positions == "localization":
        spike_depths = sorting.point_source_localizations[:, 2][keepers]
    elif drift_positions == "channel":
        spike_depths = geom[channels, 1]
    else:
        assert False

    # amplitude vectors on channel subset
    print(f"Amp vecs...")
    amp_vecs = _read_by_chunk(keep_mask, h5["denoised_ptp_amplitude_vectors"])
    amp_vecs, reassign_channel_index = waveform_util.channel_subset_by_radius(
        amp_vecs,
        channels,
        original_channel_index,
        geom,
        radius=reassign_wf_radius,
    )
    print(f"done.")
    print(f"{original_channel_index.shape=} {reassign_channel_index.shape=}")

    # static channels logic
    pitch = drift_util.get_pitch(geom)
    registered_geom = drift_util.registered_geometry(geom, motion_est=motion_est)
    registered_reassign_channel_index = waveform_util.make_channel_index(
        registered_geom, radius=reassign_wf_radius, to_torch=True
    )
    assert registered_reassign_channel_index.shape[1] == reassign_channel_index.shape[1]
    registered_original_channel_index = waveform_util.make_channel_index(
        registered_geom, radius=original_radius, to_torch=True
    )
    assert registered_original_channel_index.shape[1] == original_channel_index.shape[1]
    registered_kdtree = drift_util.KDTree(registered_geom)
    match_distance = drift_util.pdist(geom).min() / 2
    cluster_channel_index = waveform_util.make_channel_index(
        registered_geom, fit_radius
    )
    n_chans_full = len(registered_geom)
    n_chans_waveform = original_channel_index.shape[1]
    n_chans_reassign = reassign_channel_index.shape[1]
    n_chans_unit = cluster_channel_index.shape[1]
    n_spikes = keepers.size
    print("Shifts...")
    registered_depths_um = motion_est.correct_s(times_seconds, depth_um=spike_depths)
    n_pitches_shift = drift_util.get_spike_pitch_shifts(
        spike_depths,
        geom=geom,
        motion_est=motion_est,
        times_s=times_seconds,
        registered_depths_um=registered_depths_um,
        mode="round",
    )
    print("done.")

    # where a channel is not present, this has n_chans_full
    print("Chans.", end="")
    original_static_channels = drift_util.static_channel_neighborhoods(
        geom,
        channels,
        original_channel_index,
        pitch=pitch,
        n_pitches_shift=n_pitches_shift,
        registered_geom=registered_geom,
        target_kdtree=registered_kdtree,
        match_distance=match_distance,
        workers=4,
    )
    print(".", end="")
    reassign_static_channels = drift_util.static_channel_neighborhoods(
        geom,
        channels,
        reassign_channel_index,
        pitch=pitch,
        n_pitches_shift=n_pitches_shift,
        registered_geom=registered_geom,
        target_kdtree=registered_kdtree,
        match_distance=match_distance,
        workers=4,
    )
    print(".")
    static_main_channels = drift_util.static_channel_neighborhoods(
        geom,
        channels,
        np.arange(len(geom))[:, None],
        pitch=pitch,
        n_pitches_shift=n_pitches_shift,
        registered_geom=registered_geom,
        target_kdtree=registered_kdtree,
        match_distance=match_distance,
        workers=4,
    )
    static_main_channels = static_main_channels.squeeze()
    print("done.")

    # tpca embeds on channel subset
    print("Feats...")
    original_tpca_embeds = h5[tpca_feature_name]
    if in_memory:
        if interp_kind == "nearest":
            original_tpca_embeds = _read_by_chunk(keep_mask, original_tpca_embeds)
        elif interp_kind in ("normalized", "kriging", "kernel"):
            if kriging_sigma == "pitch":
                kriging_sigma = pitch
            shifts = registered_depths_um - spike_depths
            original_tpca_embeds = _interp_by_chunk(
                keep_mask,
                original_tpca_embeds,
                geom,
                original_channel_index,
                channels,
                shifts,
                registered_geom,
                original_static_channels,
                sigma=kriging_sigma,
                interp_kind=interp_kind,
            )
        else:
            assert False
        original_tpca_embeds = torch.as_tensor(original_tpca_embeds)
    else:
        assert interp_kind == "nearest"

    reassign_tpca_embeds = None
    if in_memory:
        # reassign_tpca_embeds = _channel_subset_by_chunk(
        #     keep_mask, original_tpca_embeds, channels, original_channel_index, reassign_channel_index
        # )
        reassign_tpca_embeds = waveform_util.channel_subset_by_index(
            original_tpca_embeds,
            channels,
            original_channel_index,
            reassign_channel_index,
            chunk_length=256,
        )
        h5.close()
    waveform_rank = original_tpca_embeds.shape[1]
    print("done.")

    tpca = None
    if load_tpca:
        model_dir = (
            sorting.parent_h5_path.parent / f"{sorting.parent_h5_path.stem}_models"
        )
        pipeline_pt = model_dir / "featurization_pipeline.pt"
        pipeline = torch.load(pipeline_pt)
        tpca = pipeline.transformers[0]

    return SpikeData(
        keepers=keepers,
        spike_train=sorting,
        channels=channels,
        times_seconds=times_seconds,
        times_samples=times_samples,
        waveform_rank=waveform_rank,
        n_chans_full=n_chans_full,
        n_chans_unit=n_chans_unit,
        n_chans_waveform=n_chans_waveform,
        n_chans_reassign=n_chans_reassign,
        n_spikes=n_spikes,
        motion_est=motion_est,
        original_channel_index=original_channel_index,
        reassign_channel_index=reassign_channel_index,
        registered_original_channel_index=registered_original_channel_index,
        registered_reassign_channel_index=registered_reassign_channel_index,
        cluster_channel_index=cluster_channel_index,
        amps=amps,
        amp_vecs=amp_vecs,
        original_tpca_embeds=original_tpca_embeds,
        reassign_tpca_embeds=reassign_tpca_embeds,
        original_static_channels=original_static_channels,
        reassign_static_channels=reassign_static_channels,
        static_main_channels=static_main_channels,
        registered_geom=registered_geom,
        in_memory=in_memory,
        on_device=on_device,
        pin=pin,
        tpca=tpca,
        geom=geom,
        kriging_sigma=kriging_sigma,
    )


def reassign_by_chunk(gmm, sorting):
    reassigned_labels = np.full_like(sorting.labels, -1)

    pitch = drift_util.get_pitch(gmm.data.geom)
    registered_kdtree = drift_util.KDTree(gmm.data.registered_geom.numpy(force=True))
    match_distance = drift_util.pdist(gmm.data.geom).min() / 2
    unit_ids = gmm.unit_ids()
    units = [gmm[u] for u in unit_ids]
    nu = len(units)

    with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
        embeds = h5["collisioncleaned_tpca_features"]
        nchunks = int(np.ceil(embeds.shape[0] / embeds.chunks[0]))

        for sli, *_ in tqdm(embeds.iter_chunks(), desc="Full reassign", total=nchunks):
            channels_in_chunk = sorting.channels[sli]
            chunk_times_s = sorting.times_seconds[sli]

            chunk_shifts = drift_util.get_spike_pitch_shifts(
                sorting.point_source_localizations[sli, 2],
                geom=gmm.data.geom,
                motion_est=gmm.data.motion_est,
                times_s=chunk_times_s,
            )
            static_chans = drift_util.static_channel_neighborhoods(
                gmm.data.geom,
                channels_in_chunk,
                gmm.data.reassign_channel_index,
                pitch=pitch,
                n_pitches_shift=chunk_shifts,
                registered_geom=gmm.data.registered_geom,
                target_kdtree=registered_kdtree,
                match_distance=match_distance,
                workers=4,
            )

            chunk_embeds = waveform_util.channel_subset_by_index(
                embeds[sli],
                channels_in_chunk,
                gmm.data.original_channel_index,
                gmm.data.reassign_channel_index,
            )

            static_chans = torch.tensor(static_chans, device=gmm.device)
            chunk_embeds = torch.tensor(chunk_embeds, device=gmm.device)
            chunk_times_s = torch.tensor(chunk_times_s, device=gmm.device)

            shape = (nu, len(chunk_embeds))
            divergences = np.full(shape, np.inf)

            for j, unit in enumerate(units):
                overlaps, rel_ix = unit.overlaps(static_chans)
                (which,) = torch.nonzero(overlaps >= gmm.min_overlap, as_tuple=True)
                if not which.numel():
                    continue
                _, _, res = unit.spike_badnesses(
                    chunk_times_s[which],
                    chunk_embeds[which],
                    static_chans[which],
                    overlaps=overlaps[which],
                    rel_ix=rel_ix[which],
                    kinds=(gmm.reassign_metric,),
                )
                which = which.numpy(force=True)
                divergences[j, which] = res[gmm.reassign_metric].numpy(force=True)

            # reassigned_labels[sli] = sparse_reassign(divergences, 1.0 - gmm.outlier_explained_var)

            has_match = np.isfinite(divergences).any(axis=0)
            new_labels = np.where(has_match, divergences.argmin(0), -1)
            kept = np.flatnonzero(has_match)
            outlandish = divergences[new_labels[kept], kept] >= gmm.match_threshold
            new_labels[kept[outlandish]] = -1
            reassigned_labels[sli] = new_labels

    return reassigned_labels


def reassign_by_chunk_inmem(gmm, sorting, batch_size=2 * 8192):
    reassigned_labels = np.full_like(sorting.labels, -1)

    pitch = drift_util.get_pitch(gmm.data.geom)
    registered_kdtree = drift_util.KDTree(gmm.data.registered_geom.numpy(force=True))
    match_distance = drift_util.pdist(gmm.data.geom).min() / 2
    unit_ids = gmm.unit_ids()
    units = [gmm[u] for u in unit_ids]
    nu = len(units)

    with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
        embeds = h5["collisioncleaned_tpca_features"][:]
        embeds = waveform_util.channel_subset_by_index(
            embeds,
            sorting.channels,
            gmm.data.original_channel_index,
            gmm.data.reassign_channel_index,
        )
        # embeds = _channel_subset_by_chunk(
        #     np.ones(len(embeds), dtype=bool),
        #     embeds,
        #     sorting.channels,
        #     gmm.data.original_channel_index,
        #     gmm.data.reassign_channel_index,
        #     show_progress=True,
        # )

    for start in trange(0, len(embeds), batch_size):
        # for start in trange(0, 100 * batch_size, batch_size):
        sli = slice(start, min(len(embeds), start + batch_size))
        channels_in_chunk = sorting.channels[sli]
        chunk_times_s = sorting.times_seconds[sli]

        chunk_shifts = drift_util.get_spike_pitch_shifts(
            sorting.point_source_localizations[sli, 2],
            geom=gmm.data.geom,
            motion_est=gmm.data.motion_est,
            times_s=chunk_times_s,
        )
        static_chans = drift_util.static_channel_neighborhoods(
            gmm.data.geom,
            channels_in_chunk,
            gmm.data.reassign_channel_index,
            pitch=pitch,
            n_pitches_shift=chunk_shifts,
            registered_geom=gmm.data.registered_geom,
            target_kdtree=registered_kdtree,
            match_distance=match_distance,
            workers=4,
        )

        # chunk_embeds = waveform_util.channel_subset_by_index(
        #     embeds[sli],
        #     channels_in_chunk,
        #     gmm.data.original_channel_index,
        #     gmm.data.reassign_channel_index,
        # )

        static_chans = torch.tensor(static_chans, device=gmm.device)
        chunk_embeds = torch.tensor(embeds[sli], device=gmm.device)
        chunk_times_s = torch.tensor(chunk_times_s, device=gmm.device)

        shape = (nu, len(chunk_embeds))
        divergences = np.full(shape, np.inf)

        for j, unit in enumerate(units):
            overlaps, rel_ix = unit.overlaps(static_chans)
            (which,) = torch.nonzero(overlaps >= gmm.min_overlap, as_tuple=True)
            if not which.numel():
                continue
            _, _, res = unit.spike_badnesses(
                chunk_times_s[which],
                chunk_embeds[which],
                static_chans[which],
                overlaps=overlaps[which],
                rel_ix=rel_ix[which],
                kinds=(gmm.reassign_metric,),
            )
            which = which.numpy(force=True)
            divergences[j, which] = res[gmm.reassign_metric].numpy(force=True)

        # reassigned_labels[sli] = sparse_reassign(divergences, 1.0 - gmm.outlier_explained_var)
        has_match = np.isfinite(divergences).any(axis=0)
        new_labels = np.where(has_match, divergences.argmin(0), -1)
        kept = np.flatnonzero(has_match)
        outlandish = divergences[new_labels[kept], kept] >= gmm.match_threshold
        new_labels[kept[outlandish]] = -1
        reassigned_labels[sli] = new_labels

    return reassigned_labels


# -- helpers


def cubic_interpolation_kernel(x, xeval):
    assert x.shape[0] == x.numel()
    assert xeval.shape[0] == xeval.numel()
    x = x.ravel()
    xeval = xeval.ravel()

    h = x[1] - x[0]
    n = x.numel()
    left_inds = torch.searchsorted(x, xeval, right=True) - 1
    s = (xeval - x[left_inds]) / h

    # in parentheses
    s3 = s**3
    s2 = s**2
    w0 = 0.5 * (-s3 + 2 * s2 - s)
    w3 = 0.5 * (s3 - s2)

    # main case
    inds = left_inds[:, None] + torch.arange(-1, 3, device=left_inds.device)
    weights = torch.empty(inds.shape, dtype=x.dtype, device=x.device)
    weights[:, 0] = w0
    weights[:, 1] = 0.5 * (3 * s3 - 5 * s2 + 2)
    weights[:, 2] = 0.5 * (-3 * s3 + 4 * s2 + s)
    weights[:, 3] = w3

    # points on left boundary
    left = left_inds == 0
    # inds[:, 0] += left.to(inds.dtype)  # -1 -> 0
    dw_left = torch.column_stack((-w0, 3 * w0, -3 * w0, w0))
    weights[left] += dw_left[left]

    # points on right boundary
    right = left_inds == (n - 2)
    # inds[:, 3] += right.to(inds.dtype)
    dw_right = torch.column_stack((w3, -3 * w3, 3 * w3, -w3))
    weights[right] += dw_right[right]

    # points really on the right boundary
    right = left_inds == (n - 1)
    weights[right] = 0
    weights[right, 1] = 1.0
    # inds[:, 3] += right.to(inds.dtype)

    keep = torch.logical_not(torch.logical_or(inds < 0, inds >= n))
    keep = inds_eval, _ = torch.nonzero(keep, as_tuple=True)
    inds_grid = inds[keep]
    weights = weights[keep]

    return inds_grid, inds_eval, weights


def left_interpolation_matrix(x, xeval, dim=0, kind="cubic"):
    if kind == "cubic":
        inds_grid, inds_eval, weights = cubic_interpolation_kernel(x, xeval)
    elif kind == "linear":
        inds_grid, inds_eval, weights = linear_interpolation_kernel(x, xeval)
    indices = torch.row_stack((inds_eval, inds_grid))
    left_interp_matrix = torch.sparse_coo_tensor(
        indices,
        weights,
        size=(xeval.shape[dim], x.shape[dim]),
    )
    # can't csr, no grads in torch for csr, but they have coo grad
    # might be better to do dense?
    # left_interp_matrix = left_interp_matrix.to_sparse_csr()
    left_interp_matrix = left_interp_matrix.to_dense()
    return left_interp_matrix


def linear_interpolation_kernel(x, xeval):
    assert x.shape[0] == x.numel()
    assert xeval.shape[0] == xeval.numel()
    x = x.ravel()
    xeval = xeval.ravel()

    h = x[1] - x[0]
    n = x.numel()
    left_inds = torch.searchsorted(x, xeval, right=True) - 1
    s = (xeval - x[left_inds]) / h

    # main case
    inds = left_inds[:, None] + torch.arange(0, 2, device=left_inds.device)
    weights = torch.empty(inds.shape, dtype=x.dtype, device=x.device)
    weights[:, 0] = 1.0 - s
    weights[:, 1] = s

    # points on left boundary
    left = left_inds == -1
    weights[left, 0] = 0
    weights[left, 1] = 1

    # points on right boundary
    pass

    # points really on the right boundary
    right = left_inds == (n - 1)
    weights[right] = 0
    weights[right, 1] = 1.0
    # inds[:, 3] += right.to(inds.dtype)

    keep = torch.logical_not(torch.logical_or(inds < 0, inds >= n))
    keep = inds_eval, _ = torch.nonzero(keep, as_tuple=True)
    inds_grid = inds[keep]
    weights = weights[keep]

    return inds_grid, inds_eval, weights


def _read_by_chunk(mask, dataset, show_progress=True):
    """
    mask : boolean array of shape dataset.shape[:1]
    dataset : chunked h5py.Dataset
    """
    out = np.empty((mask.sum(), *dataset.shape[1:]), dtype=dataset.dtype)
    n = 0
    chunks = dataset.iter_chunks()
    if show_progress:
        chunks = tqdm(
            chunks,
            total=int(np.ceil(dataset.shape[0] / dataset.chunks[0])),
            desc=dataset.name,
        )
    for sli, *_ in chunks:
        m = np.flatnonzero(mask[sli])
        nm = m.size
        if not nm:
            continue
        x = dataset[sli][m]
        # x = dataset[np.arange(sli.start, sli.stop)[m]]
        out[n : n + nm] = x
        n += nm
    return out


def _interp_by_chunk(
    mask,
    dataset,
    geom,
    channel_index,
    channels,
    shifts,
    registered_geom,
    target_channels,
    device=None,
    sigma=20.0,
    show_progress=True,
    dtype=torch.float,
    interp_kind="kriging",
):
    """
    mask : boolean array of shape dataset.shape[:1]
    dataset : chunked h5py.Dataset
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    chunk_size = dataset.chunks[0]

    # build needed torch data
    source_geom = torch.as_tensor(geom, dtype=dtype, device=device)
    source_geom = F.pad(source_geom, (0, 0, 0, 1), value=torch.nan)
    target_geom = torch.as_tensor(registered_geom, dtype=dtype, device=device)
    target_geom = F.pad(target_geom, (0, 0, 0, 1), value=torch.nan)
    shifts = torch.as_tensor(shifts, dtype=dtype)
    target_channels = torch.as_tensor(target_channels)
    channel_index = torch.as_tensor(channel_index)
    channels = torch.as_tensor(channels)
    zeros = torch.zeros((chunk_size,), device=device, dtype=dtype)

    # same batching logic as read by chunk
    out = np.empty((mask.sum(), *dataset.shape[1:]), dtype=dataset.dtype)
    n = 0
    chunks = dataset.iter_chunks()
    if show_progress:
        chunks = tqdm(
            chunks,
            total=int(np.ceil(dataset.shape[0] / chunk_size)),
            desc=f"Interpolated {dataset.name}",
        )

    source_invs = None
    if interp_kind.startswith("kriging"):
        source_pos = source_geom[channel_index].to(device)
        source_kernel = log_rbf(source_pos[None])
        source_kernel = source_kernel.exp_()
        source_invs = torch.linalg.pinv(source_kernel)

    for sli, *_ in chunks:
        m = np.flatnonzero(mask[sli])
        nm = m.size
        if not nm:
            continue
        x = dataset[sli][m]

        # to torch
        x = torch.from_numpy(x).to(device)
        source_channels = channel_index[channels[n : n + nm]].to(device)
        # print(f"{(source_channels == len(geom)).sum(1)=}")
        # print(f"{torch.isnan(x).all(1).sum(1)=}")
        # print(f"{(target_channels[n : n + nm] == len(registered_geom)).sum(1)=}")
        source_shifts = shifts[n : n + nm].to(device)
        source_shifts = torch.column_stack((zeros[:nm], source_shifts))
        source_pos = source_geom[source_channels] + source_shifts.unsqueeze(1)
        target_pos = target_geom[target_channels[n : n + nm].to(device)]
        ski = source_invs[channels[n:n+nm]] if interp_kind == "kriging" else None
        x = kernel_interpolate(x, source_pos, target_pos, sigma=sigma, allow_destroy=True, kind=interp_kind, source_kernel_invs=ski)

        # x = dataset[np.arange(sli.start, sli.stop)[m]]
        out[n : n + nm] = x.numpy(force=True)
        n += nm

    return out


def kernel_interpolate(
    features,
    source_pos,
    target_pos,
    source_kernel_invs=None,
    sigma=20.0,
    allow_destroy=False,
    kind="normalized",
    out=None,
):
    # geoms should be nan-padded here.
    # build kernel
    kernel = log_rbf(source_pos, target_pos, sigma)
    if kind == "normalized":
        kernel = F.softmax(kernel, dim=1)
    elif kind.startswith("kriging"):
        kernel = kernel.exp_()
        kernel = source_kernel_invs @ kernel
        if kind == "kriging_normalized":
            kernel = kernel / kernel.sum(1, keepdim=True)
    elif kind == "kernel":
        kernel = kernel.exp_()
    else:
        assert False
    torch.nan_to_num(kernel, out=kernel)

    # and apply...
    features = torch.nan_to_num(features, out=features if allow_destroy else None)
    err_str = f"before {features.min()=} {features.max()=}"
    features = torch.bmm(features, kernel, out=out)
    if features.abs().max() > 1000:
        err_str += "\nfafter {features.min()=} {features.max()=}"
        raise ValueError(err_str)

    # nan-ify nonexistent chans
    needs_nan = torch.isnan(target_pos).all(2).unsqueeze(1)
    needs_nan = needs_nan.broadcast_to(features.shape)
    features[needs_nan] = torch.nan

    return features


def log_rbf(source_pos, target_pos=None, sigma=None):
    if target_pos is None:
        target_pos = source_pos
    kernel = torch.cdist(source_pos, target_pos)
    kernel = kernel.square_().mul_(-1.0 / (2 * sigma**2))
    torch.nan_to_num(kernel, nan=-torch.inf, out=kernel)
    return kernel


def svd_lstsq(AA, BB, tol=1e-5, driver='gesvd'):
    U, S, Vh = torch.linalg.svd(AA, full_matrices=False, driver=driver)
    Spinv = torch.zeros_like(S)
    Spinv[S>tol] = 1/S[S>tol]
    UhBB = U.adjoint() @ BB
    if Spinv.ndim!=UhBB.ndim:
        Spinv = Spinv.unsqueeze(-1)
    SpinvUhBB = Spinv * UhBB
    return Vh.adjoint() @ SpinvUhBB


def eigh_lstsq(AA, BB, tol=1e-5):
    S, Q = torch.linalg.eigh(AA)
    valid = S > tol
    S[valid] = 1.0 / S[valid]
    return torch.einsum("nuv,nv,nwv,nwx->nux", Q, S, Q, BB)


def _channel_subset_by_chunk(
    mask,
    dataset,
    channels,
    original_channel_index,
    new_channel_index,
    show_progress=False,
):
    out = np.empty(
        (mask.sum(), dataset.shape[1], new_channel_index.shape[1]), dtype=dataset.dtype
    )
    n = 0
    iterator = dataset.iter_chunks()
    if show_progress:
        nchunks = int(np.ceil(dataset.shape[0] / dataset.chunks[0]))
        iterator = tqdm(iterator, total=nchunks)

    for sli, *_ in iterator:
        m = np.flatnonzero(mask[sli])
        nm = m.size
        if not nm:
            continue
        x = dataset[sli][m]
        # x = dataset[np.arange(sli.start, sli.stop)[m]]
        out[n : n + nm] = waveform_util.channel_subset_by_index(
            x, channels[n : n + nm], original_channel_index, new_channel_index
        )
        n += nm
    return out


def mad(x, axis=None, keepdims=False):
    x = x - np.median(x, axis=axis, keepdims=True)
    np.abs(x, out=x)
    return np.median(x, axis=axis, keepdims=keepdims)


def sparse_reassign(divergences, match_threshold=None, batch_size=512, return_csc=False):
    # this uses CSC-specific tricks to do fast argmax per column
    if not divergences.nnz:
        return np.full(divergences.shape[0], -1)

    # see scipy csc argmin/argmax for reference here. this is just numba-ing
    # a special case of that code which has a python hot loop.
    divergences = divergences.tocsc()
    errs = divergences # if match_threshold is None else divergences.copy()
    nz_lines = np.flatnonzero(np.diff(divergences.indptr))
    offset = errs.data.max() + 1
    errs.data -= offset
    assignments = np.full(errs.shape[1], -1)
    hot_argmin_loop(assignments, nz_lines, errs.indptr, errs.data, errs.indices)
    errs.data += offset
    if match_threshold is None:
        if return_csc:
            return errs, assignments
        return assignments

    # we want sparse 0s to mean infinite err, and divs>thresh
    # to be infinite as well. right now, [0, 1] with 0 as best
    # subtract M to go to [-M, -M + 1].
    outlandish = divergences[assignments[nz_lines], nz_lines] >= match_threshold
    assignments[nz_lines[outlandish]] = -1

    if return_csc:
        return errs, assignments
    return assignments


@numba.njit(
    numba.void(
        numba.int64[:], numba.int64[:], numba.int32[:], numba.float32[:], numba.int32[:]
    ),
    error_model="numpy",
    nogil=True,
)
def hot_argmin_loop(assignments, nz_lines, indptr, data, indices):
    for i in nz_lines:
        p = indptr[i]
        q = indptr[i + 1]
        ix = indices[p:q]
        dx = data[p:q]
        assignments[i] = ix[dx.argmin()]
