from __future__ import print_function
import numpy as np
import caiman as cm
import pylab as pl
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import itertools as itt
import scipy.misc as misc
import os
import glob
import re
import warnings
# import sparse
import time
import re
import functools
from scipy import ndimage as ndi
from scipy import signal as sig
from scipy.ndimage.measurements import center_of_mass
from scipy.stats import sem
from caiman.source_extraction.cnmf import cnmf
from caiman import motion_correction, components_evaluation
from skvideo import io as sio
# from matplotlib_venn import venn2
from collections import deque, OrderedDict
from decimal import Decimal
# import numericbtree as nbtree


def save_video(movpath, fname_mov_orig, fname_mov_rig, fname_AC, fname_ACbf, dsratio):
    mov_orig = np.load(fname_mov_orig, mmap_mode='r')
    mov_rig = np.load(fname_mov_rig, mmap_mode='r')
    mov_ac = np.load(fname_AC, mmap_mode='r')
    mov_acbf = np.load(fname_ACbf, mmap_mode='r')
    vw = sio.FFmpegWriter(
        movpath,
        inputdict={
            '-framerate': '30'
        },
        outputdict={
            '-r': '30'
        })
    for fidx in range(0, mov_orig.shape[0], dsratio):
        print("writing frame: " + str(fidx))
        fm_orig = mov_orig[fidx, :, :] * 255
        fm_rig = mov_rig[fidx, :, :] * 255
        fm_acbf = mov_acbf[fidx, :, :] * 255
        fm_ac = mov_ac[fidx, :, :] * 255
        fm = np.concatenate([
            np.concatenate([fm_orig, fm_rig], axis=1),
            np.concatenate([fm_acbf, fm_ac], axis=1)],
            axis=0)
        vw.writeFrame(fm)
    vw.close()


def align_across_session(a1, a2, fn_mov_rig1, fn_mov_rig2):
    mov_rig1 = np.load(fn_mov_rig1, mmap_mode='r')
    mov_rig2 = np.load(fn_mov_rig2, mmap_mode='r')
    mov_rig1_mean = np.mean(mov_rig1, axis=0)
    mov_rig2_mean = np.mean(mov_rig2, axis=0)
    cross_corr = sig.fftconvolve(mov_rig1_mean, mov_rig2_mean, mode='same')
    maximum = np.unravel_index(np.argmax(cross_corr), mov_rig1_mean.shape)
    midpoints = mov_rig1_mean.shape / 2
    shifts = maximum - midpoints
    a2.reshape((mov_rig1_mean.shape, -1))
    a2 = np.roll(a2, shifts, (0, 1))
    return a1, a2, shifts


def estimate_overlap(a1, a2, dims=None, dist_cutoff=5, method='max', search_range=5, restrict_search=True):
    if np.ndim(a1) < 3:
        a1 = a1.reshape(np.append(dims, [-1]), order='F')
        a2 = a2.reshape(np.append(dims, [-1]), order='F')
    centroids_a1 = np.zeros((a1.shape[2], 2))
    centroids_a2 = np.zeros((a2.shape[2], 2))
    dist_centroids = np.zeros((a1.shape[2], a2.shape[2]))
    for idca1, ca1 in enumerate(centroids_a1):
        centroids_a1[idca1, :] = center_of_mass(a1[:, :, idca1])
    for idca2, ca2 in enumerate(centroids_a2):
        centroids_a2[idca2, :] = center_of_mass(a2[:, :, idca2])
    for idxs, dist in np.ndenumerate(dist_centroids):
        print("calculating distance for pair: " + str(idxs))
        ca1 = centroids_a1[idxs[0]]
        ca2 = centroids_a2[idxs[1]]
        dist_centroids[idxs] = np.sqrt((ca1[0] - ca2[0]) ** 2 + (ca1[1] - ca2[1]) ** 2)
    dist_min0 = np.tile(np.min(dist_centroids, axis=1), (dist_centroids.shape[1], 1)).transpose()
    dist_min0 = dist_centroids == dist_min0
    dist_min1 = np.tile(np.min(dist_centroids, axis=0), (dist_centroids.shape[0], 1))
    dist_min1 = dist_centroids == dist_min1
    dist_mask = np.logical_and(dist_min0, dist_min1)
    dist_cut = dist_centroids < dist_cutoff
    dist_mask = np.logical_and(dist_mask, dist_cut)
    correlations = np.zeros((a1.shape[2], a2.shape[2], 3))
    if method:
        min_idxs = list()
        if restrict_search:
            min_dist_a1 = np.argmin(dist_centroids, axis=1)
            min_dist_a2 = np.argmin(dist_centroids, axis=0)
            for ida1, ida2 in enumerate(min_dist_a1):
                min_idxs.append((ida1, ida2))
            for ida2, ida1 in enumerate(min_dist_a2):
                min_idxs.append((ida1, ida2))
        for idxs, corr in np.ndenumerate(correlations[:, :, 0]):
            if min_idxs and idxs not in min_idxs:
                correlations[idxs + (0,)] = -1
                continue
            else:
                print("calculating correlation for pair: " + str(idxs))
                if dist_centroids[idxs] < dist_cutoff:
                    if method == 'max':
                        search_dims = (search_range * 2 + 1, search_range * 2 + 1)
                        corr_temp = np.zeros(search_dims)
                        for id_shift, corr_shift in np.ndenumerate(corr_temp):
                            shift = tuple(ish - search_range for ish in id_shift)
                            a1_temp = np.roll(a1[:, :, idxs[0]], shift[0], axis=0)
                            a1_temp = np.roll(a1_temp, shift[1], axis=1).flatten()
                            a2_temp = a2[:, :, idxs[1]].flatten()
                            corr_temp[id_shift] = np.corrcoef(a1_temp, a2_temp)[0, 1]
                        max_shift = np.unravel_index(np.argmax(corr_temp), search_dims)
                        correlations[idxs + (0,)] = np.max(corr_temp)
                        correlations[idxs + (1,)] = max_shift[0]
                        correlations[idxs + (2,)] = max_shift[1]
                    elif method == 'plain':
                        a1_temp = a1[:, :, idxs[0]].flatten()
                        a2_temp = a2[:, :, idxs[1]].flatten()
                        correlations[idxs + (0,)] = np.corrcoef(a1_temp, a2_temp)[0, 1]
                    else:
                        print("Unrecognized method!")
                else:
                    correlations[idxs + (0,)] = -1
    nua1 = a1.shape[2]
    nua2 = a2.shape[2]
    ovlp = np.sum(dist_mask)
    return dist_centroids, dist_mask, correlations, (nua1, nua2, ovlp)


def infer_map_old(*args):
    ovlp = np.eye(args[0].shape[0], dtype=bool)
    res = list()
    for idmask, mask in enumerate(args):
        if ovlp.shape[1] != mask.shape[0]:
            mask = mask.T
            if ovlp.shape[1] == mask.shape[0]:
                warnings.warn("dimension mismatch, using transpose of matrix " + str(idmask))
            else:
                warnings.warn("dimension mismatch, skipping matrix " + str(idmask))
                continue
        new_ovlp = np.zeros((ovlp.shape[0], mask.shape[1]), dtype=bool)
        pairs_ovlp = np.nonzero(ovlp)
        pairs_mask = np.nonzero(mask)
        for idpovlp, povlp in enumerate(pairs_ovlp[1]):
            if povlp in pairs_mask[0]:
                idpmask = pairs_mask[1][np.where(pairs_mask[0] == povlp)]
                new_ovlp[pairs_ovlp[0][idpovlp], idpmask] = True
        ovlp = new_ovlp
    return ovlp


def calculate_centroids_old(*args):
    ndims = np.array(np.ndim(a) for a in args)
    if np.any(ndims < 3):
        raise AssertionError("not a spatial matrix. reshape first!")
    nunits = tuple(a.shape[-1] for a in args)
    centroids = list()
    for ida, cur_a in enumerate(args):
        print ("calculating centroids for matrix " + str(ida))
        cur_centroid = np.zeros((nunits[ida], 2))
        for idu, u in enumerate(cur_centroid):
            cur_centroid[idu, :] = center_of_mass(cur_a[:, :, idu])
        centroids.append(cur_centroid)
    return centroids


def calculate_centroids_distance_old(*args, **kwargs):
    ndims = np.array(np.ndim(a) for a in args)
    if np.any(ndims < 3):
        raise AssertionError("not a spatial matrix. reshape first!")
    nunits = tuple(a.shape[-1] for a in args)
    dims = set([a.shape[0:2] for a in args])
    if len(dims) > 1:
        warnings.warn("inputs dimensions mismatch. using the dimensions of first spatial matrix")
    dims = dims.pop()
    centroids = kwargs.get('cent_in', list())
    if not centroids:
        centroids = calculate_centroids_old(*args)
    tile = kwargs.get('tile', None)
    if tile:
        cent0 = np.linspace(0, dims[0], np.ceil(dims[0] * 2.0 / tile[0]))
        cent1 = np.linspace(0, dims[1], np.ceil(dims[1] * 2.0 / tile[1]))
        coords = np.empty(shape=(len(nunits), 0))
        for cent in itt.product(cent0, cent1):
            print ("center: " + str(cent))
            centroids_inrange = list()
            for cur_centroids in centroids:
                cur_inrange = cur_centroids[:, 0] > (cent[0] - np.ceil(tile[0] / 2.0))
                cur_inrange = np.logical_and(cur_inrange, cur_centroids[:, 0] < (cent[0] + np.ceil(tile[0] / 2.0)))
                cur_inrange = np.logical_and(cur_inrange, cur_centroids[:, 1] > (cent[1] - np.ceil(tile[1] / 2.0)))
                cur_inrange = np.logical_and(cur_inrange, cur_centroids[:, 1] < (cent[1] + np.ceil(tile[1] / 2.0)))
                centroids_inrange.append(np.nonzero(cur_inrange)[0])
            cur_coords = np.empty(shape=(len(nunits), 0))
            for pair_inrange in itt.product(*centroids_inrange):
                pair_inrange = np.array(pair_inrange).reshape((len(nunits), -1))
                cur_coords = np.append(cur_coords, pair_inrange, axis=1)
            coords = np.hstack((coords, cur_coords))
        dist_centroids = sparse.COO(coords, data=np.array((-1,) * coords.shape[1]), shape=nunits)
    else:
        dist_centroids = np.zeros(nunits, dtype=np.float32)
    dist_it = np.nditer(dist_centroids, flags=['multi_index'], op_flags=['readwrite'])
    print ("calculating centroids distance with shape: " + str(dist_centroids.shape))
    while not dist_it.finished:
        idx = dist_it.multi_index
        cur_centroids = np.array([centroids[ida][idu, :] for ida, idu in enumerate(idx)])
        midpoint = np.tile(np.mean(cur_centroids, axis=0), (len(cur_centroids), 1))
        dist_it[0] = np.sum(np.sqrt(np.sum((cur_centroids - midpoint) ** 2, axis=1)))
        dist_it.iternext()
    return dist_centroids


def estimate_threshold(*args):
    ndims = np.array(np.ndim(a) for a in args)
    if np.any(ndims < 3):
        raise AssertionError("not a spatial matrix. reshape first!")
    nunits = tuple(a.shape[-1] for a in args)
    if len(nunits) < 3:
        warnings.warn("less than 3 matrix provided. returning default threshold")
        return 5
    dist_list = []
    args_sh = deque(args)
    args_sh.rotate(-1)
    print ("threshold estimation start")
    for s0, s1 in zip(args, args_sh):
        dist_list.append(calculate_centroids_distance(s0, s1))
    thres = 0
    while thres < 100:
        map_list = deque()
        for ids, (s0, s1) in enumerate(zip(args, args_sh)):
            cur_map = calculate_map_old(s0, s1, dist_in=dist_list[ids], threshold=thres + 1)
            map_list.append(cur_map)
        infer_list = []
        for _ in range(len(map_list)):
            cur_infer = infer_map(*map_list)
            infer_list.append(cur_infer)
            map_list.rotate(1)
        if any(not np.array_equal(*np.nonzero(im)) for im in infer_list):
            print ("estimated threshold: " + str(thres))
            break
        else:
            thres += 1
    return thres


def calculate_map_old(*args, **kwargs):
    ndims = np.array(np.ndim(a) for a in args)
    if np.any(ndims < 3):
        raise AssertionError("not a spatial matrix. reshape first!")
    nunits = tuple(a.shape[-1] for a in args)
    method = kwargs.get('method', 'perunit')
    thres = kwargs.get('threshold', None)
    centroids = kwargs.get('cent_in', list())
    dist_centroids = kwargs.get('dist_in', np.zeros(nunits))
    dist_map = np.ones(nunits, dtype=bool)
    if not dist_centroids.any():
        if not centroids:
            dist_centroids = calculate_centroids_distance_old(*args)
        else:
            dist_centroids = calculate_centroids_distance_old(*args, cent_in=centroids)
    if not thres:
        thres = estimate_threshold(*args)
    thres = len(nunits) * np.sqrt(thres ** 2 / (2 - 2 * np.cos(2 * np.pi / len(nunits))))
    print ("using threshold: " + str(thres))
    for axis in range(dist_centroids.ndim):
        cur_dist = dist_centroids.swapaxes(0, axis)
        cur_map = np.zeros_like(dist_map).swapaxes(0, axis)
        if method == 'perunit':
            for uid, unit in enumerate(cur_dist):
                pid = np.unravel_index(np.argmin(unit), cur_map.shape[1:])
                cur_map[(uid,) + pid] = True
        elif method == 'perpair':
            cur_min = np.argmin(cur_dist, axis=0)
            min_it = np.nditer(cur_min, flags=['multi_index'])
            while not min_it.finished:
                cur_map[(min_it[0],) + min_it.multi_index] = True
                min_it.iternext()
        else:
            raise ValueError("unrecognized method")
        cur_thres = cur_dist < thres
        cur_map = np.logical_and(cur_map, cur_thres)
        cur_map = cur_map.swapaxes(0, axis)
        dist_map = np.logical_and(dist_map, cur_map)
    return dist_map


def batch_load_spatial_component(animalpath):
    sa = OrderedDict()
    for dirname, subdirs, files in os.walk(animalpath):
        if files:
            if os.path.isfile(dirname + os.sep + 'cnm.npz'):
                dirnamelist = dirname.split(os.sep)
                cur_a = np.load(dirname + os.sep + 'cnm.npz')['A']
                dims = np.load(dirname + os.sep + 'cnm.npz')['dims']
                cur_a = np.reshape(cur_a, np.append(dims, (-1,)), order='F')
                sessionid = 's' + dirnamelist[-2]
                cur_a = xr.DataArray(
                    cur_a,
                    coords={
                        'ay': range(cur_a.shape[0]),
                        'ax': range(cur_a.shape[1]),
                        'unitid': range(cur_a.shape[2])
                    },
                    dims=('ay', 'ax', 'unitid'),
                    name=sessionid
                )
                sa[sessionid] = cur_a
                print ("loading: " + dirname)
            else:
                print ("cnm not found in folder: " + dirname + " proceed")
        else:
            print("empty folder: " + dirname + " proceed")
    return sa


def batch_calculate_centroids(sa):
    cent_dict = OrderedDict()
    for sid, a in sa.items():
        cent_dict[sid] = calculate_centroids(a)
    return cent_dict


def calculate_centroids(a):
    print ("calculating centroids for " + a.name)
    centroids = np.zeros((a.shape[2], 2))
    for idu, u in enumerate(centroids):
        centroids[idu, :] = center_of_mass(a.values[:, :, idu])
    centroids = xr.DataArray(
        centroids.T,
        coords={
            'centloc': ['cy', 'cx'],
            'unitid': range(a.shape[2])
        },
        dims=('centloc', 'unitid'),
        name=a.name
    )
    return centroids


def batch_calculate_centroids_distances(cent, dims, tile=None):
    t_start = time.time()
    dist_dict = OrderedDict()
    for comblen in range(2, len(cent) + 1):
        for comb in itt.combinations(cent.items(), comblen):
            sidset = tuple([itpair[0] for itpair in comb])
            centset = dict(comb)
            print ("calculating distance of centroids for sessions:"
                   + str(sidset))
            dist = calculate_centroids_distance(centset, dims, tile)
            dist_dict[sidset] = dist
    print ("total running time:" + str(time.time() - t_start))
    return dist_dict


def calculate_centroids_distance(centroids, dims, tile=None):
    nunits = tuple(a.shape[-1] for a in centroids.values())
    print ("number of units: " + str(nunits))
    if not tile:
        tile = dims
    centy = np.linspace(0, dims[0], np.ceil(dims[0] * 2.0 / tile[0]) + 1)
    centx = np.linspace(0, dims[1], np.ceil(dims[1] * 2.0 / tile[1]) + 1)
    dy = np.ceil(tile[0] / 2.0)
    dx = np.ceil(tile[1] / 2.0)
    coords = list()
    centroids_y = dict([(sid, sa.sel(centloc='cy').values) for (sid, sa) in centroids.items()])
    centroids_x = dict([(sid, sa.sel(centloc='cx').values) for (sid, sa) in centroids.items()])
    for cy, cx in itt.product(centy, centx):
        centroids_inrange = list()
        for sid, cur_cent in centroids.items():
            inrangey = np.logical_and(centroids_y[sid] > (cy - dy), centroids_y[sid] < (cy + dy))
            inrangex = np.logical_and(centroids_x[sid] > (cx - dx), centroids_x[sid] < (cx + dx))
            inrange = np.logical_and(inrangex, inrangey)
            # centroids_inrange.append(inrange.where(inrange).unitid)
            centroids_inrange.append(np.nonzero(inrange)[0].tolist())
        for pair in itt.product(*centroids_inrange):
            coords.append(pair)
    mulidx = pd.MultiIndex.from_tuples(coords, names=centroids.keys())
    mulidx = mulidx.drop_duplicates()
    print ("subsetting " + str(mulidx.shape[0]) + " pairs")
    dist = np.array([])
    for idx in mulidx:
        cur_centroids = np.array([centroids[centroids.keys()[ids]][:, idu] for ids, idu in enumerate(idx)])
        midpoint = np.tile(np.mean(cur_centroids, axis=0), (len(cur_centroids), 1))
        dist = np.append(dist, np.sum(np.sqrt(np.sum((cur_centroids - midpoint) ** 2, axis=1))))
    # dist_it = np.nditer(dist_centroids, flags=['multi_index'], op_flags=['readwrite'])
    # print ("calculating centroids distance with shape: " + str(dist_centroids.shape))
    # while not dist_it.finished:
    #     idx = dist_it.multi_index
    #     cur_centroids = np.array([centroids[ida][idu, :] for ida, idu in enumerate(idx)])
    #     midpoint = np.tile(np.mean(cur_centroids, axis=0), (len(cur_centroids), 1))
    #     dist_it[0] = np.sum(np.sqrt(np.sum((cur_centroids - midpoint) ** 2, axis=1)))
    #     dist_it.iternext()
    return pd.Series(dist, mulidx, name='distance')


def batch_calculate_map(dist):
    map_dict = OrderedDict()
    for sname, d in dist.items():
        map_dict[sname] = calculate_map(d)
    return map_dict


def calculate_map(dist):
    minidx = dist.index
    for sname in dist.index.names:
        cur_minidx = dist.groupby(level=sname).apply(lambda d: d.argmin())
        minidx = minidx.intersection(cur_minidx)
    dist_map = dist[minidx]
    return dist_map


def initialize_meta_map(map_dict):
    snames = list(map_dict.keys()[np.argmax(map(len, map_dict.keys()))])
    meta_all = pd.DataFrame(pd.concat([m.reset_index() for m in map_dict.values()], ignore_index=True))
    meta_all = reset_meta_map(meta_all, snames)
    # meta_all = threshold_meta_map(meta_all, 10)
    meta_all = meta_all.sort_values('nsession', ascending=False)
    update_meta_map(meta_all, snames)
    return meta_all, snames


def reset_meta_map(meta_all, snames=None):
    if not snames:
        snames = meta_all.sessions
    meta_all[snames] = meta_all[snames].apply(pd.to_numeric, downcast='integer')
    meta_all['nsession'] = meta_all[snames].count(axis=1)
    meta_all['missing'] = [[]] * len(meta_all)
    meta_all['conflict'] = [[]] * len(meta_all)
    meta_all['conflict_with'] = [[]] * len(meta_all)
    meta_all['match_score'] = np.nan
    meta_all['missing_score'] = np.nan
    meta_all['conflict_score'] = np.nan
    meta_all['score'] = np.nan
    meta_all['active'] = True
    return meta_all


def update_meta_map(meta_all, snames):
    row_count = 0
    for cur_idx, cur_entry in meta_all.iterrows():
        row_count += 1
        if row_count % 100 == 0:
            print ("iteration: {} out of {}".format(row_count, meta_all.shape[0]))
        cur_sname = cur_entry[snames].dropna().index
        cur_block = meta_all[meta_all['active']]
        cur_block = cur_block[pd.DataFrame(meta_all == cur_entry)[cur_sname].any(axis=1)]
        sub_block = cur_block[cur_block['nsession'] <= cur_entry['nsession']]
        sub_block_nm = ~sub_block.apply(lambda r: (r == cur_entry).replace(~r.isnull(), True), axis=1)[cur_sname]
        cur_nm = sub_block[sub_block_nm.apply(lambda r: r.any(), axis=1)]
        cur_conf = ~cur_block.apply(lambda r: (r == cur_entry).replace(~r.isnull(), True), axis=1)[cur_sname]
        cur_conf = cur_block[cur_conf.apply(lambda r: r.any(), axis=1)]
        cur_conf = set(cur_conf.index) - {cur_idx}
        cur_exp = set(frozenset(cmb) for slen in range(2, len(cur_sname)) for cmb in itt.combinations(cur_sname, slen))
        cur_block_sname = set(sub_block.apply(lambda r: frozenset(r[cur_sname].dropna().index), axis=1))
        cur_ms = cur_exp - cur_block_sname
        meta_all.set_value(cur_idx, 'missing', meta_all.loc[cur_idx, 'missing'] + [tuple(m) for m in cur_ms])
        meta_all.set_value(cur_idx, 'conflict', meta_all.loc[cur_idx, 'conflict'] + list(cur_nm.index))
        meta_all.loc[cur_conf, 'conflict_with'] = meta_all.loc[cur_conf, 'conflict_with'].apply(lambda r: r + [cur_idx])
        match_score = sub_block.apply(lambda r: r == cur_entry, axis=1)[cur_sname].values.sum()
        missing_score = len(list(itt.chain.from_iterable(meta_all.loc[cur_idx, 'missing'])))
        conflict_score = np.sum(sub_block_nm.values)
        score = match_score - conflict_score
        meta_all.set_value(cur_idx, 'match_score', match_score)
        meta_all.set_value(cur_idx, 'missing_score', missing_score)
        meta_all.set_value(cur_idx, 'conflict_score', conflict_score)
        meta_all.set_value(cur_idx, 'score', score)


def threshold_meta_map(meta_all, threshold):
    thres = meta_all['nsession'].apply(lambda x: x * np.sqrt(threshold ** 2 / (2 - 2 * np.cos(2 * np.pi / x))))
    return meta_all[meta_all['distance'] < thres]


def resolve_conflicts(meta_all, snames):
    fcutoff = 50
    # meta_all['nconflict'] = meta_all['conflict_with'].apply(lambda l: np.sum(l))
    # meta_all = meta_all.sort_values('nconflict', ascending=False)
    conf_pairs = meta_all[meta_all['conflict_with'].apply(bool)]['conflict_with']
    conf_list = [[list(conf_pairs.index).index(c) for c in p] for p in conf_pairs]
    nconf = len(conf_list)
    conf_tree = nbtree.NBTree(nconf)
    remv_exp_last = 0
    print ("processing conflict list with length {}".format(nconf))
    for level in range(fcutoff, nconf):
        t = time.time()
        comb_exp = np.sum([misc.comb(level, posslen) for posslen in range(fcutoff, level + 1)])
        remv_exp = comb_exp - remv_exp_last*2
        remv_exp_last = comb_exp
        nremoved = 0
        nsearched = 0
        print ("processing level: " + str(level))
        print ("expected removal: " + str(remv_exp))
        for nd in conf_tree.get_nodes(level=level, unpack=True):
            print ("searched: {0}, removed: {1}".format(nsearched, nremoved), end='\r')
            if nremoved >= remv_exp:
                break
            if nd % 2:
                nsearched += 1
                fal = np.array(conf_tree.path_to_node(nd)) % 2
                if np.sum(fal) >= fcutoff:
                    conf_tree.remove_subtree(nd)
                    nremoved += 1
        print ("process time: {} s".format(time.time() - t))
    for conid1, pair in enumerate(conf_list):
        for conid2 in pair:
            conidl = min(conid1, conid2)
            conidh = max(conid1, conid2)
            print ("processing pair:" + str((conidl, conidh)))
            for ndl in conf_tree.get_nodes(level=conidl + 1, unpack=True):
                if not ndl / 2:
                    for ndh in conf_tree.children(ndl).get_nodes(level=conidh + 1, unpack=True):
                        if not ndh / 2:
                            conf_tree.remove_subtree(ndh)
    return conf_tree


def cut_nodes(meta_all, cutoff):
    conf_pairs = meta_all[meta_all['conflict_with'].apply(bool)]['conflict_with']
    conf_list = [[list(conf_pairs.index).index(c) for c in p] for p in conf_pairs]
    conf_tree = nbtree.NBTree(len(conf_list))
    beg_node = conf_tree.leaves(0).begin()
    end_node = conf_tree.leaves(0).end()
    cur_node = beg_node
    processed = 0
    removed = 0
    while cur_node < end_node:
        if processed % 1000 == 0:
            print("processing {0:.15E} th node. processed: {1}, removed: {2}, left: {3:.15E}".format(
                cur_node - beg_node, processed, removed, Decimal(end_node - cur_node)))
        cur_node = conf_tree.get_nodes(level=conf_tree._depth, unpack=True).next()
        path = conf_tree.path_to_node(cur_node)
        pathodd = [p % 2 for p in path]
        iodd = -1
        try:
            for _ in range(cutoff):
                iodd = pathodd.index(1, iodd + 1)
        except ValueError:
            pass
        if iodd > 0:
            conf_tree.remove_subtree(path[iodd])
            removed += 1
        processed += 1
    return conf_tree


def group_meta_map(meta_all):
    meta_grp = meta_all.copy()
    grplen = meta_grp['group'].apply(lambda l: len(l))
    for cur_idx, cur_entry in meta_grp[grplen > 1].iterrows():
        cand = cur_entry['group']
        cand_scr = np.array([meta_grp.loc[can, 'match_score'] if not meta_grp.loc[can, 'conflict_score']
                             else 0 for can in cand])
        meta_grp.set_value(cur_idx, 'group', [cand[ic] for ic in np.where(cand_scr == cand_scr.min())[0]])
    # meta_grp.loc[grplen == 1, 'group'] = meta_grp.loc[grplen == 1, 'group'].apply(lambda l: l[0])
    return meta_grp


def subset_data_by_list(data, column, tlist):
    return data[data[column].apply(lambda l: bool(set(l) & set(tlist)))]


def subset_data_by_map(meta_all, snames):
    snames = list(snames)
    snames_all = filter(lambda x: re.match('s[0-9]+$', x), meta_all.columns)
    snames_nan = list(set(snames_all) - set(snames))
    meta_null = meta_all[snames_all].isnull()
    satisfied = meta_all[meta_null.apply(lambda r: ~r[snames].any() and r[snames_nan].all(), axis=1)]
    satisfied.sessions = snames
    return satisfied


def infer_meta_map(meta_all, snames=None):
    if not snames:
        try:
            snames = meta_all.sessions
        except AttributeError:
            snames = filter(lambda x: re.match('s[0-9]+$', x), meta_all.columns)
    infer_list = []
    for nsession in range(3, len(snames) + 1):
        for sessions_infer in itt.combinations(snames, nsession):
            print ("inferring " + str(sessions_infer))
            infer = infer_map(meta_all, sessions_infer)
            infer_list.append(infer)
    pair_list = [subset_data_by_map(meta_all, s) for s in itt.combinations(snames, 2)]
    meta_infer = pd.concat(infer_list + pair_list, ignore_index=True)
    meta_infer = meta_infer.reindex_axis(meta_all.columns, axis=1)
    meta_infer = reset_meta_map(meta_infer, snames)
    meta_infer.sessions = snames
    return meta_infer


def infer_map(meta_all, snames):
    meta_pair_list = [subset_data_by_map(meta_all, s) for s in itt.combinations(snames, 2)]
    meta_infer = filter(lambda m: snames[0] in m.sessions, meta_pair_list)[0]
    for cur_on in snames:
        cur_maps = filter(lambda m: cur_on in m.sessions, meta_pair_list)
        for cur_m in cur_maps:
            meta_infer = extend_map(meta_infer, cur_m, cur_on)
    meta_infer.sessions = snames
    return meta_infer


def extend_map(mapx, mapy, on):
    if not hasattr(on, '__iter__'):
        on = [on]
    try:
        inter = mapx[on].reset_index().merge(mapy[on].reset_index(), on=on)
    except KeyError:
        inter = pd.DataFrame()
    extended = pd.DataFrame()
    ext_sessions = list(set(mapx.sessions).union(set(mapy.sessions)))
    for inter_idx, inter_row in inter.iterrows():
        sx = mapx.loc[inter_row.loc['index_x'], mapx.sessions]
        sy = mapy.loc[inter_row.loc['index_y'], mapy.sessions]
        extrow = pd.concat([sx, sy]).drop_duplicates()
        if len(extrow) <= len(ext_sessions):
            extended = extended.append(extrow, ignore_index=True)
    extended.sessions = ext_sessions
    return extended


def generate_summary(mapdict, sadict):
    if len(mapdict) != len(sadict):
        raise ValueError("length of mappings and spatial components mismatch!")
    exp_meta = pd.concat(mapdict, names=['animal', 'original_index']).reset_index()
    snames = filter(lambda x: re.match('s[0-9]+$', x), exp_meta.columns)
    exp_meta['sessions'] = exp_meta[snames].apply(lambda r: tuple(r[r.notnull()].index.tolist()), axis=1)
    exp_meta = exp_meta.groupby('animal').apply(group_by_session)
    summary = exp_meta.groupby(['animal', 'grouping_by_session', 'sessions']).size().reset_index(name='count')
    grouping_dict = dict(summary.groupby(['animal', 'grouping_by_session']).groups.keys())
    for cur_anm, cur_sa in sadict.items():
        for cur_session, cur_s in cur_sa.items():
            summary = summary.append(pd.Series({
                'animal': cur_anm,
                'grouping_by_session': grouping_dict[cur_anm],
                'sessions': (cur_session,),
                'count': len(cur_s.unitid)
            }), ignore_index=True)
    return summary, exp_meta


def generate_overlap(summary, denominator = 'each'):
    summary_map = summary[summary['sessions'].apply(lambda x: len(x)) > 1]
    calculation = functools.partial(calculate_overlap, summary=summary)
    overlaps = summary_map.groupby(['animal', 'sessions']).apply(calculation)
    return overlaps


def plot_overlaps(overlaps, subset):
    overlaps['group'].replace({'shock': 'neutral', 'non-shock': 'valence'}, inplace=True)
    overlap_mean = overlaps.groupby(['mappings', 'on', 'group']).apply(np.mean).unstack('group').reset_index()
    overlap_std = overlaps.groupby(['mappings', 'on', 'group']).apply(np.std).unstack('group').reset_index()
    overlap_mean_sub = overlap_mean.loc[
                       overlap_mean.mappings.apply(lambda i: set(i) <= subset), :].set_index(['mappings', 'on'])
    overlap_std_sub = overlap_std.loc[
                      overlap_std.mappings.apply(lambda i: set(i) <= subset), :].set_index(['mappings', 'on'])
    overlap_mean_sub.plot(kind='bar', yerr=overlap_std_sub)


def calculate_overlap(mapping, summary, on='each'):
    if isinstance(mapping, pd.DataFrame):
        if len(mapping) > 1:
            raise ValueError("can only handle one mapping at a time!")
        else:
            mapping = mapping.iloc[0]
    if on == 'total':
        pass
    elif on == 'each':
        summary_single = summary[summary['sessions'].apply(lambda x: len(x)) == 1]
        summary_single = summary_single[summary_single['animal'] == mapping['animal']]
        summary_single.loc[:, 'sessions'] = summary_single['sessions'].apply(set)
        cur_snames = set(mapping['sessions'])
        cur_dev = summary_single[summary_single['sessions'].apply(lambda s: s <= cur_snames)]
        cur_dev.loc[:, 'sessions'] = cur_dev['sessions'].apply(tuple)
    else:
        cur_dev = summary.loc[(summary['sessions'] == on) & (summary['animal'] == mapping['animal'])]
    overlap = pd.DataFrame()
    overlap['group'] = cur_dev['group']
    overlap['animal'] = cur_dev['animal']
    overlap['mappings'] = [mapping['sessions']] * len(cur_dev)
    overlap['on'] = cur_dev['sessions']
    overlap['freq'] = mapping['count'] * 1.0 / cur_dev['count']
    return overlap


def group_by_session(group):
    cur_snames = set()
    for ss in group.sessions.unique():
        cur_snames.update(ss)
    group['grouping_by_session'] = [tuple(sorted(cur_snames))] * len(group)
    return group


def plot_spatial(alist, idlist=None, dims=None, ax=None, cmaplist=None):
    if not idlist:
        idlist = []
        for cur_a in alist:
            idlist.append(np.arange(cur_a.shape[-1]))
    if not cmaplist:
        cmaplist = []
        cmaplist = ['gray'] * len(alist)
    if not ax:
        ax = pl.gca()
        ax.set_facecolor('white')
    for ida, a in enumerate(alist):
        if np.ndim(a) < 3:
            a = a.reshape(np.append(dims, [-1]))
        for idx in idlist[ida]:
            ax.imshow(np.ma.masked_equal(a[:, :, idx], 0), alpha=0.5, cmap=cmaplist[ida])
    return ax


def plot_dist_vs_corr(dist_centroids, dist_mask, correlations, savepath=None, suffix=''):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    y = dist_centroids.flatten()
    x = correlations[:, :, 0].flatten()
    mask = dist_mask.flatten()
    outpoints = ax.scatter(x=x[~mask], y=y[~mask], c='b')
    inpoints = ax.scatter(x=x[mask], y=y[mask], c='r')
    ax.set_xlabel("correlation")
    ax.set_ylabel("distance of centroids")
    ax.legend((inpoints, outpoints),
              ('masked', 'residule'))
    if savepath:
        fig.savefig(savepath + 'dist_vs_corr' + suffix + '.svg')


def plot_dist_hist(dist_centroids, cut_range=(1, 20, 1), restrict_min=True):
    cuts = range(cut_range[0], cut_range[1], cut_range[2])
    ncut = len(cuts)
    fig = pl.figure(figsize=(6 * 5.5, ncut * 5.5))
    for id_cut, cut in enumerate(cuts):
        dist_filtered = np.ones_like(dist_centroids)
        if restrict_min:
            dist_min0 = np.tile(np.min(dist_centroids, axis=1), (dist_centroids.shape[1], 1)).transpose()
            dist_min0 = dist_centroids == dist_min0
            dist_min1 = np.tile(np.min(dist_centroids, axis=0), (dist_centroids.shape[0], 1))
            dist_min1 = dist_centroids == dist_min1
            dist_filtered = np.logical_and(dist_min0, dist_min1)
        dist_filtered = np.logical_and(dist_filtered, dist_centroids < cut)
        nmatch0 = np.sum(dist_filtered, axis=1)
        nmatch1 = np.sum(dist_filtered, axis=0)
        plt1 = fig.add_subplot(ncut, 2, id_cut * 2 + 1)
        plt2 = fig.add_subplot(ncut, 2, id_cut * 2 + 2)
        plt1.hist(nmatch0, bins=30, range=(0, 30))
        plt2.hist(nmatch1, bins=30, range=(0, 30))
        plt1.set_title("matches for first spatial matrix. dist_cutoff: " + str(cut))
        plt2.set_title("matches for second spatial matrix. dist_cutoff: " + str(cut))
        plt1.set_xlabel("number of matches")
        plt2.set_xlabel("number of matches")
    fig.savefig('/home/phild/dist_hist_min.svg', bboxinches='tight', dpi=300)


def plot_venn(sets, setlabels, savepath=''):
    pl.rcParams.update({'font.size': '19'})
    fig = pl.figure()
    nsets = len(sets)
    for setid, cur_set in enumerate(sets):
        ax = fig.add_subplot(nsets, 1, setid + 1)
        v = venn2(cur_set, set_labels=setlabels[setid], ax=ax)
        ratio0 = cur_set[2] / float(cur_set[0])
        ratio1 = cur_set[2] / float(cur_set[1])
        ratiomean = cur_set[2] / np.mean([cur_set[0], cur_set[1]])
        ratiosum = cur_set[2] / float(cur_set[0] + cur_set[1] - cur_set[2])
        a = setlabels[setid][0]
        b = setlabels[setid][1]
        pl.text(
            1, 0.8,
            r'$\frac{' + a + r' \cap ' + b + r'}{' + a + r'} = ' + '{:.3}'.format(ratio0) + r'$',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        pl.text(
            1, 0.6,
            r'$\frac{' + a + r' \cap ' + b + r'}{' + b + r'} = ' + '{:.3}'.format(ratio1) + r'$',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        pl.text(
            1, 0.4,
            r'$\frac{' + a + r' \cap ' + b + r'}{mean(' + a + r', ' + b + r')} = ' + '{:.3}'.format(ratiomean) + r'$',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        pl.text(
            1, 0.2,
            r'$\frac{' + a + r' \cap ' + b + r'}{' + a + r' \cup ' + b + r'} = ' + '{:.3}'.format(ratiosum) + r'$',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(a + ' vs ' + b)


def plot_components(a, c, dims, savepath=''):
    try:
        a = a.reshape(np.append(dims, -1), order='F')
    except NotImplementedError:
        a = a.toarray().reshape(np.append(dims, -1), order='F')
    if savepath:
        pl.ioff()
    for cmp_id, temp_sig in enumerate(c):
        fig = pl.figure()
        ax_a = fig.add_subplot(211)
        ax_c = fig.add_subplot(212)
        ax_a.imshow(a[:, :, cmp_id])
        ax_c.plot(temp_sig)
        fig.suptitle("component " + str(cmp_id))
        if savepath:
            fig.savefig(savepath + "component_" + str(cmp_id) + '.svg')
            print("saving component " + str(cmp_id))
    pl.ion()


def process_data(dpath, movpath, pltpath, roi):
    params_movie = {
        'niter_rig': 1,  # maximum number of iterations rigid motion correction,
        # in general is 1. 0 will quickly initialize a template with the first frames
        'max_shifts': (20, 20),  # maximum allow rigid shift
        'splits_rig': 28,  # for parallelization split the movies in  num_splits chuncks across time
        #  if none all the splits are processed and the movie is saved
        'num_splits_to_process_rig': None,
        # intervals at which patches are laid out for motion correction
        'strides': (48, 48),
        # overlap between pathes (size of patch strides+overlaps)
        'overlaps': (24, 24),
        'splits_els': 28,  # for parallelization split the movies in  num_splits chuncks across time
        # if none all the splits are processed and the movie is saved
        'num_splits_to_process_els': [14, None],
        'upsample_factor_grid': 4,  # upsample factor to avoid smearing when merging patches
        # maximum deviation allowed for patch with respect to rigid
        # shift
        'max_deviation_rigid': 3,
        'p': 1,  # order of the autoregressive system
        'merge_thresh': 0.9,  # merging threshold, max correlation allowed
        'rf': 40,  # half-size of the patches in pixels. rf=25, patches are 50x50
        'stride_cnmf': 20,  # amounpl.it of overlap between the patches in pixels
        'K': 4,  # number of components per patch
        # if dendritic. In this case you need to set init_method to
        # sparse_nmf
        'is_dendrites': False,
        'init_method': 'greedy_roi',
        'gSig': [10, 10],  # expected half size of neurons
        'alpha_snmf': None,  # this controls sparsity
        'final_frate': 30
    }
    if not dpath.endswith(os.sep):
        dpath = dpath + os.sep
    if not os.path.isfile(dpath + 'mc.npz'):
        # start parallel
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)
        dpattern = 'msCam*.avi'
        dlist = sorted(glob.glob(dpath + dpattern),
                       key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        if not dlist:
            print("No data found in the specified folder: " + dpath)
            return
        else:
            vdlist = list()
            for vname in dlist:
                vdlist.append(sio.vread(vname, as_grey=True))
            mov_orig = cm.movie(np.squeeze(np.concatenate(vdlist, axis=0))).astype(np.float32)
            # column correction
            meanrow = np.mean(np.mean(mov_orig, 0), 0)
            addframe = np.tile(meanrow, (mov_orig.shape[1], 1))
            mov_cc = mov_orig - np.tile(addframe, (mov_orig.shape[0], 1, 1))
            mov_cc = mov_cc - np.min(mov_cc)
            # filter
            mov_ft = mov_cc.copy()
            for fid, fm in enumerate(mov_cc):
                mov_ft[fid] = ndi.uniform_filter(fm, 2) - ndi.uniform_filter(fm, 40)
            mov_orig = (mov_orig - np.min(mov_orig)) / (np.max(mov_orig) - np.min(mov_orig))
            mov_ft = (mov_ft - np.min(mov_ft)) / (np.max(mov_ft) - np.min(mov_ft))
            np.save(dpath + 'mov_orig', mov_orig)
            np.save(dpath + 'mov_ft', mov_ft)
            del mov_orig, dlist, vdlist, mov_ft
            mc_data = motion_correction.MotionCorrect(
                dpath + 'mov_ft.npy', 0,
                dview=dview, max_shifts=params_movie['max_shifts'],
                niter_rig=params_movie['niter_rig'], splits_rig=params_movie['splits_rig'],
                num_splits_to_process_rig=params_movie['num_splits_to_process_rig'],
                strides=params_movie['strides'], overlaps=params_movie['overlaps'],
                splits_els=params_movie['splits_els'],
                num_splits_to_process_els=params_movie['num_splits_to_process_els'],
                upsample_factor_grid=params_movie['upsample_factor_grid'],
                max_deviation_rigid=params_movie['max_deviation_rigid'],
                shifts_opencv=True, nonneg_movie=False, roi=roi)
            mc_data.motion_correct_rigid(save_movie=True)
            mov_rig = cm.load(mc_data.fname_tot_rig)
            np.save(dpath + 'mov_rig', mov_rig)
            np.savez(dpath + 'mc',
                     fname_tot_rig=mc_data.fname_tot_rig,
                     templates_rig=mc_data.templates_rig,
                     shifts_rig=mc_data.shifts_rig,
                     total_templates_rig=mc_data.total_template_rig,
                     max_shifts=mc_data.max_shifts,
                     roi=mc_data.roi)
            del mov_rig
    else:
        print("motion correction data already exist. proceed")
    if not os.path.isfile(dpath + "cnm.npz"):
        # start parallel
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)
        fname_tot_rig = np.array_str(np.load(dpath + 'mc.npz')['fname_tot_rig'])
        mov, dims, T = cm.load_memmap(fname_tot_rig)
        mov = np.reshape(mov.T, [T] + list(dims), order='F')
        cnm = cnmf.CNMF(
            n_processes, k=params_movie['K'], gSig=params_movie['gSig'], merge_thresh=params_movie['merge_thresh'],
            p=params_movie['p'], dview=dview, Ain=None, rf=params_movie['rf'], stride=params_movie['stride_cnmf'],
            memory_fact=1, method_init=params_movie['init_method'], alpha_snmf=params_movie['alpha_snmf'],
            only_init_patch=True,
            gnb=1, method_deconvolution='oasis')
        cnm = cnm.fit(mov)
        # Cn = cm.local_correlations(mov_orig, swap_dim=False)
        idx_comp, idx_comp_bad = components_evaluation.estimate_components_quality(
            cnm.C + cnm.YrA, np.reshape(mov, dims + (T,), order='F'), cnm.A, cnm.C, cnm.b,
            cnm.f, params_movie['final_frate'], Npeaks=10, r_values_min=.7, fitness_min=-40, fitness_delta_min=-40
        )
        # visualization.plot_contours(cnm.A.tocsc()[:, idx_comp], Cn)
        A2 = cnm.A.tocsc()[:, idx_comp]
        C2 = cnm.C[idx_comp]
        cnm = cnmf.CNMF(
            n_processes, k=A2.shape, gSig=params_movie['gSig'], merge_thresh=params_movie['merge_thresh'],
            p=params_movie['p'],
            dview=dview, Ain=A2, Cin=C2, f_in=cnm.f, rf=None, stride=None, method_deconvolution='oasis')
        cnm = cnm.fit(mov)
        idx_comp, idx_comp_bad = components_evaluation.estimate_components_quality(
            cnm.C + cnm.YrA, np.reshape(mov, dims + (T,), order='F'), cnm.A, cnm.C, cnm.b,
            cnm.f, params_movie['final_frate'], Npeaks=10, r_values_min=.75, fitness_min=-50, fitness_delta_min=-50
        )
        cnm.A = cnm.A.tocsc()[:, idx_comp]
        cnm.C = cnm.C[idx_comp]
        # visualization.plot_contours(cnm.A.tocsc()[:, idx_comp], Cn)
        cm.cluster.stop_server()
        cnm.A = (cnm.A - np.min(cnm.A)) / (np.max(cnm.A) - np.min(cnm.A))
        cnm.C = (cnm.C - np.min(cnm.C)) / (np.max(cnm.C) - np.min(cnm.C))
        cnm.b = (cnm.b - np.min(cnm.b)) / (np.max(cnm.b) - np.min(cnm.b))
        cnm.f = (cnm.f - np.min(cnm.f)) / (np.max(cnm.f) - np.min(cnm.f))
        np.savez(dpath + 'cnm', A=cnm.A.todense(), C=cnm.C, b=cnm.b, f=cnm.f, YrA=cnm.YrA, sn=cnm.sn, dims=dims)
        # AC = (cnm.A.dot(cnm.C)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        # ACmin = np.min(AC)
        # ACmax = np.max(AC)
        # AC = (AC - ACmin) / (ACmax - ACmin)
        # np.save(dpath + 'AC', AC)
        # del AC, ACmax, ACmin
        # ACbf = (cnm.A.dot(cnm.C) + cnm.b.dot(cnm.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        # ACbfmin = np.min(ACbf)
        # ACbfmax = np.max(ACbf)
        # ACbf = (ACbf - ACbfmin) / (ACbfmax - ACbfmin)
        # np.save(dpath + 'ACbf', ACbf)
        # del ACbf, ACbfmax, ACbfmin
        # plist = dpath.split(os.sep)
        # vname = plist[-3] + '_' + plist[-2] + '_result.mp4'
        # save_video(
        #     movpath+vname, dpath + 'mov_orig.npy', dpath + 'mov_rig.npy', dpath + 'AC.npy',
        #     dpath + 'ACbf.npy', dsratio=3)
    else:
        print("cnm data already exist. proceed")
        # os.remove(dpath + 'mov_orig.npy')
        # os.remove(dpath + 'mov_ft.npy')
        # os.remove(mc_data.fname_tot_rig)
        # os.remove(dpath+'mov_rig.npy')
        # os.remove(dpath + 'AC.npy')
        # os.remove(dpath + 'ACbf.npy')
    try:
        A = cnm.A
        C = cnm.C
        dims = dims
    except NameError:
        A = np.load(dpath + 'cnm.npz')['A']
        C = np.load(dpath + 'cnm.npz')['C']
        dims = np.load(dpath + 'cnm.npz')['dims']
    plot_components(A, C, dims, pltpath)


def batch_process_data(animal_path, movroot, pltroot, roi):
    for dirname, subdirs, files in os.walk(animal_path):
        if files:
            dirnamelist = dirname.split(os.sep)
            dname = dirnamelist[-3] + '_' + dirnamelist[-2]
            movpath = movroot + os.sep + dname
            pltpath = pltroot + os.sep + dname
            if not os.path.exists(movpath):
                os.mkdir(movpath)
            if not os.path.exists(pltpath):
                os.mkdir(pltpath)
            movpath = movpath + os.sep
            pltpath = pltpath + os.sep
            process_data(dirname, movpath, pltpath, roi)
        else:
            print("empty folder: " + dirname + " proceed")

        # if __name__ == '__main__':
        # a1 = np.load('/media/share/Denise/Wired Valence/Wired Valence Organized Data/MS101/4/H11_M52_S45/cnm.npz')['A']
        # a2 = np.load('/media/share/Denise/Wired Valence/Wired Valence Organized Data/MS101/5/H11_M41_S56/cnm.npz')['A']
        # dims = np.load('/media/share/Denise/Wired Valence/Wired Valence Organized Data/MS101/4/H11_M52_S45/cnm.npz')['dims']
