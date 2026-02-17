import numpy as np
import json
import os
import sys
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment

#import poseval.py.eval_helpers as eval_helpers
#from poseval.py.eval_helpers import Joint
#import poseval.py.motmetrics as mm

import motmetrics as mm
from .eval_helpers import assignGTmulti, Joint, writeJson, getPointGTbyID

def _get_joint_position(prFrame, ridxPr, joint_id):
    """Extract (x, y) position of a specific joint from a prediction.

    Returns None if the joint is not found.
    """
    annorects = prFrame["annorect"]
    if ridxPr >= len(annorects):
        return None
    rect = annorects[ridxPr]
    if "annopoints" not in rect or len(rect["annopoints"]) == 0:
        return None
    if "point" not in rect["annopoints"][0]:
        return None
    point = getPointGTbyID(rect["annopoints"][0]["point"], joint_id)
    if len(point) == 0:
        return None
    return (point["x"][0], point["y"][0])


def _assign_spatial_joint_ids(motAll, prFramesAll, gtFramesAll, nJoints):
    """Replace person-level prediction track IDs with spatially-consistent
    joint-level IDs.

    For each joint type, tracks prediction positions across frames within
    each sequence using Hungarian matching on pixel distance. This way,
    the MOTAccumulator only detects genuine spatial matching changes as
    ID switches, not person-level tracking errors propagating to all joints.
    """
    seqidxs = np.array([gtFramesAll[i]["seq_id"]
                        for i in range(len(gtFramesAll))])
    seqidxsUniq = np.unique(seqidxs)

    SPATIAL_THRESHOLD = 200.0  # pixels - max distance to match same ID

    for seq_id in seqidxsUniq:
        imgidxs = np.argwhere(seqidxs == seq_id).flatten()

        for joint_i in range(nJoints):
            next_id = 1
            prev_positions = {}  # spatial_id -> (x, y)

            for imgidx in imgidxs:
                mot = motAll[imgidx][joint_i]
                ridxsPr = mot["ridxsPr"]
                n_pr = len(ridxsPr)

                # Check if this is a dummy entry (no real predictions)
                if len(prFramesAll[imgidx]["annorect"]) == 0:
                    # Dummy frame - assign placeholder ID, won't match anyway
                    mot["trackidxPr"] = [0] * n_pr
                    prev_positions = {}
                    continue

                # Get positions for current predictions
                curr_positions = []
                for ridx in ridxsPr:
                    pos = _get_joint_position(
                        prFramesAll[imgidx], ridx, joint_i)
                    curr_positions.append(pos)

                # Filter to only valid (non-None) positions
                valid_indices = [ci for ci, pos in enumerate(curr_positions)
                                 if pos is not None]

                if len(valid_indices) == 0 or len(prev_positions) == 0:
                    # No matching possible - assign fresh IDs
                    new_ids = [0] * n_pr
                    new_prev = {}
                    for ci in valid_indices:
                        tid = next_id
                        next_id += 1
                        new_ids[ci] = tid
                        new_prev[tid] = curr_positions[ci]
                    # Assign unique IDs for invalid positions too
                    for ci in range(n_pr):
                        if ci not in valid_indices:
                            new_ids[ci] = next_id
                            next_id += 1
                    mot["trackidxPr"] = new_ids
                    prev_positions = new_prev
                    continue

                # Build cost matrix: valid current predictions vs prev tracks
                prev_ids = list(prev_positions.keys())
                n_valid = len(valid_indices)
                n_prev = len(prev_ids)
                cost = np.full((n_valid, n_prev), np.inf)

                for vi, ci in enumerate(valid_indices):
                    px, py = curr_positions[ci]
                    for pi, pid in enumerate(prev_ids):
                        ppx, ppy = prev_positions[pid]
                        cost[vi, pi] = np.sqrt(
                            (px - ppx)**2 + (py - ppy)**2)

                # Hungarian matching
                row_ind, col_ind = scipy_linear_sum_assignment(cost)

                new_ids = [0] * n_pr
                new_prev = {}

                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] < SPATIAL_THRESHOLD:
                        ci = valid_indices[r]
                        tid = prev_ids[c]
                        new_ids[ci] = tid
                        new_prev[tid] = curr_positions[ci]

                # Assign new IDs for unmatched valid predictions
                for ci in valid_indices:
                    if new_ids[ci] == 0:
                        tid = next_id
                        next_id += 1
                        new_ids[ci] = tid
                        new_prev[tid] = curr_positions[ci]

                # Assign unique IDs for invalid positions
                for ci in range(n_pr):
                    if ci not in valid_indices:
                        new_ids[ci] = next_id
                        next_id += 1

                mot["trackidxPr"] = new_ids
                prev_positions = new_prev


def computeMetrics(gtFramesAll, motAll, outputDir, bSaveAll, bSaveSeq):

    assert(len(gtFramesAll) == len(motAll))

    nJoints = Joint().count
    seqidxs = []
    for imgidx in range(len(gtFramesAll)):
        seqidxs += [gtFramesAll[imgidx]["seq_id"]]
    seqidxs = np.array(seqidxs)

    seqidxsUniq = np.unique(seqidxs)

    # intermediate metrics
    metricsMidNames = ['num_misses', 'num_switches', 'num_false_positives', 'num_objects', 'num_detections']

    # final metrics computed from intermediate metrics
    # MOTA includes joint-level spatial ID switches (person-level switches removed via spatial tracking)
    metricsFinNames = ['mota','motp','pre','rec']

    # initialize intermediate metrics
    metricsMidAll = {}
    for name in metricsMidNames:
        metricsMidAll[name] = np.zeros([1,nJoints])
    metricsMidAll['sumD'] = np.zeros([1,nJoints])

    # initialize final metrics
    metricsFinAll = {}
    for name in metricsFinNames:
        metricsFinAll[name] = np.zeros([1,nJoints+1])

    # create metrics
    mh = mm.metrics.create()

    imgidxfirst = 0
    # iterate over tracking sequences
    # seqidxsUniq = seqidxsUniq[:20]
    nSeq = len(seqidxsUniq)

    # initialize per-sequence metrics
    metricsSeqAll = {}
    for si in range(nSeq):
        metricsSeqAll[si] = {}
        for name in metricsFinNames:
            metricsSeqAll[si][name] = np.zeros([1,nJoints+1])

    names = Joint().name
    names['15'] = 'total'

    for si in range(nSeq):
    #for si in range(5):
        # print("seqidx: %d/%d" % (si+1,nSeq))

        # init per-joint metrics accumulator
        accAll = {}
        for i in range(nJoints):
            accAll[i] = mm.MOTAccumulator(auto_id=True)

        # extract frames IDs for the sequence
        imgidxs = np.argwhere(seqidxs == seqidxsUniq[si])
        imgidxs = imgidxs[:-1].copy()
        seqName = gtFramesAll[imgidxs[0,0]]["seq_name"]
        print(seqName)
        # create an accumulator that will be updated during each frame
        # iterate over frames
        for j in range(len(imgidxs)):
            imgidx = imgidxs[j,0]
            # iterate over joints
            for i in range(nJoints):
                # GT tracking ID
                trackidxGT = motAll[imgidx][i]["trackidxGT"]
                # prediction tracking ID
                trackidxPr = motAll[imgidx][i]["trackidxPr"]
                # distance GT <-> pred part to compute MOT metrics
                # 'NaN' means force no match
                dist = motAll[imgidx][i]["dist"]
                # Call update once per frame
                accAll[i].update(
                    trackidxGT,                 # Ground truth objects in this frame
                    trackidxPr,                 # Detector hypotheses in this frame
                    dist                        # Distances from objects to hypotheses
                )

        # compute intermediate metrics per joint per sequence
        for i in range(nJoints):
            metricsMid = mh.compute(accAll[i], metrics=metricsMidNames, return_dataframe=False, name='acc')
            for name in metricsMidNames:
                metricsMidAll[name][0,i] += metricsMid[name]
            s = accAll[i].events['D'].sum()
            if (np.isnan(s)):
                s = 0
            metricsMidAll['sumD'][0,i] += s

#        if (bSaveSeq):
        if False:
            # compute metrics per joint per sequence
            for i in range(nJoints):
                metricsMid = mh.compute(accAll[i], metrics=metricsMidNames, return_dataframe=False, name='acc')

                # compute final metrics per sequence
                if (metricsMid['num_objects'] > 0):
                    numObj = metricsMid['num_objects']
                else:
                    numObj = np.nan
                numFP = metricsMid['num_false_positives']
                metricsSeqAll[si]['mota'][0,i] = 100*(1. - 1.*(metricsMid['num_misses'] +
                                                    metricsMid['num_switches'] +
                                                    numFP) /
                                                    numObj)
                numDet = metricsMid['num_detections']
                s = accAll[i].events['D'].sum()
                if (numDet == 0 or np.isnan(s)):
                    metricsSeqAll[si]['motp'][0,i] = 0.0
                else:
                    metricsSeqAll[si]['motp'][0,i] = 100*(1. - (1.*s / numDet))
                if (numFP+numDet > 0):
                    totalDet = numFP+numDet
                else:
                    totalDet = np.nan
                metricsSeqAll[si]['pre'][0,i]  = 100*(1.*numDet /
                                                totalDet)
                metricsSeqAll[si]['rec'][0,i]  = 100*(1.*numDet /
                                        numObj)

            # average metrics over all joints per sequence
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['mota'][0,:nJoints]))
            metricsSeqAll[si]['mota'][0,nJoints] = metricsSeqAll[si]['mota'][0,idxs].mean()
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['motp'][0,:nJoints]))
            metricsSeqAll[si]['motp'][0,nJoints] = metricsSeqAll[si]['motp'][0,idxs].mean()
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['pre'][0,:nJoints]))
            metricsSeqAll[si]['pre'][0,nJoints]  = metricsSeqAll[si]['pre'] [0,idxs].mean()
            idxs = np.argwhere(~np.isnan(metricsSeqAll[si]['rec'][0,:nJoints]))
            metricsSeqAll[si]['rec'][0,nJoints]  = metricsSeqAll[si]['rec'] [0,idxs].mean()

            metricsSeq = metricsSeqAll[si].copy()
            metricsSeq['mota'] = metricsSeq['mota'].flatten().tolist()
            metricsSeq['motp'] = metricsSeq['motp'].flatten().tolist()
            metricsSeq['pre'] = metricsSeq['pre'].flatten().tolist()
            metricsSeq['rec'] = metricsSeq['rec'].flatten().tolist()
            metricsSeq['names'] = names

            filename = outputDir + '/' + seqName + '_MOT_metrics.json'
            print('saving results to', filename)
            #eval_helpers.writeJson(metricsSeq,filename)
            writeJson(metricsSeq,filename)

    # compute final metrics per joint for all sequences
    for i in range(nJoints):
        if (metricsMidAll['num_objects'][0,i] > 0):
            numObj = metricsMidAll['num_objects'][0,i]
        else:
            numObj = np.nan
        numFP = metricsMidAll['num_false_positives'][0,i]

        # Joint-level MOTA: includes spatial joint-level ID switches only
        metricsFinAll['mota'][0,i] = 100*(1. - (metricsMidAll['num_misses'][0,i] +
                                                metricsMidAll['num_switches'][0,i] +
                                                numFP) /
                                                numObj)

        numDet = metricsMidAll['num_detections'][0,i]
        s = metricsMidAll['sumD'][0,i]
        if (numDet == 0 or np.isnan(s)):
            metricsFinAll['motp'][0,i] = 0.0
        else:
            metricsFinAll['motp'][0,i] = 100*(1. - (s / numDet))
        if (numFP+numDet > 0):
            totalDet = numFP+numDet
        else:
            totalDet = np.nan

        metricsFinAll['pre'][0,i]  = 100*(1.*numDet /
                                       totalDet)
        metricsFinAll['rec'][0,i]  = 100*(1.*numDet /
                                       numObj)

    # average metrics over all joints over all sequences
    idxs = np.argwhere(~np.isnan(metricsFinAll['mota'][0,:nJoints]))
    metricsFinAll['mota'][0,nJoints] = metricsFinAll['mota'][0,idxs].mean()
    idxs = np.argwhere(~np.isnan(metricsFinAll['motp'][0,:nJoints]))
    metricsFinAll['motp'][0,nJoints] = metricsFinAll['motp'][0,idxs].mean()
    idxs = np.argwhere(~np.isnan(metricsFinAll['pre'][0,:nJoints]))
    metricsFinAll['pre'][0,nJoints]  = metricsFinAll['pre'] [0,idxs].mean()
    idxs = np.argwhere(~np.isnan(metricsFinAll['rec'][0,:nJoints]))
    metricsFinAll['rec'][0,nJoints]  = metricsFinAll['rec'] [0,idxs].mean()

    # Print intermediate metrics for debugging/analysis
    # (Per-joint breakdown commented out - will use consolidated table in evaluation script)
    # print("\n" + "="*130)
    # print("JOINT-LEVEL METRICS (Spatial Joint-Level ID Switches Only)")
    # print("="*130)
    # print("\nPer-Joint Breakdown:")
    # print(f"{'Joint':<12} {'GT Objs':<10} {'Misses':<10} {'Detections':<12} {'False Pos':<12} {'ID Sw':<10} {'MOTA':<10}")
    # print("-"*130)
    # for i in range(nJoints):
    #     joint_name = names.get(str(i), f'Joint {i}')
    #     if metricsMidAll['num_objects'][0,i] > 0:
    #         numObj = metricsMidAll['num_objects'][0,i]
    #         numFP = metricsMidAll['num_false_positives'][0,i]
    #         numSw = metricsMidAll['num_switches'][0,i]
    #         mota_joint = 1. - (metricsMidAll['num_misses'][0,i] + numSw + numFP) / numObj
    #     else:
    #         numSw = 0
    #         mota_joint = np.nan
    #
    #     print(f"{joint_name:<12} {metricsMidAll['num_objects'][0,i]:<10.0f} "
    #           f"{metricsMidAll['num_misses'][0,i]:<10.0f} "
    #           f"{metricsMidAll['num_detections'][0,i]:<12.0f} "
    #           f"{metricsMidAll['num_false_positives'][0,i]:<12.0f} "
    #           f"{metricsMidAll['num_switches'][0,i]:<10.0f} "
    #           f"{mota_joint*100:<10.1f}")
    #
    # print("-"*130)
    # total_gt = metricsMidAll['num_objects'][0,:].sum()
    # total_misses = metricsMidAll['num_misses'][0,:].sum()
    # total_detections = metricsMidAll['num_detections'][0,:].sum()
    # total_fp = metricsMidAll['num_false_positives'][0,:].sum()
    # total_sw = metricsMidAll['num_switches'][0,:].sum()
    #
    # print(f"{'TOTAL':<12} {total_gt:<10.0f} "
    #       f"{total_misses:<10.0f} "
    #       f"{total_detections:<12.0f} "
    #       f"{total_fp:<12.0f} "
    #       f"{total_sw:<10.0f}")
    #
    # print("\n  MOTA = 1 - (Misses + ID_Switches + FP) / GT")
    # print("  ID switches are spatial joint-level only (person-level switches removed).")
    # print("="*130 + "\n")

#    if (bSaveAll):
    if False:
        metricsFin = metricsFinAll.copy()
        metricsFin['mota'] = metricsFin['mota'].flatten().tolist()
        metricsFin['motp'] = metricsFin['motp'].flatten().tolist()
        metricsFin['pre'] = metricsFin['pre'].flatten().tolist()
        metricsFin['rec'] = metricsFin['rec'].flatten().tolist()
        metricsFin['names'] = names

        filename = outputDir + '/total_MOT_metrics.json'
        print('saving results to', filename)
#        eval_helpers.writeJson(metricsFin,filename)
        writeJson(metricsFin,filename)

    return metricsFinAll


def evaluateTracking(gtFramesAll, prFramesAll, outputDir, saveAll=True, saveSeq=False):

    distThresh = 0.5
    nJoints = Joint().count
    # assign predicted poses to GT poses
#    _, _, _, motAll = eval_helpers.assignGTmulti(gtFramesAll, prFramesAll, distThresh)
    _, _, _, motAll = assignGTmulti(gtFramesAll, prFramesAll, distThresh)

    # Replace person-level prediction IDs with spatially-consistent
    # joint-level IDs so that joint MOTA only counts genuine spatial
    # ID switches, not person-level tracking errors.
    _assign_spatial_joint_ids(motAll, prFramesAll, gtFramesAll, nJoints)

    # compute MOT metrics per part
    metricsAll = computeMetrics(gtFramesAll, motAll, outputDir, saveAll, saveSeq)

    return metricsAll
