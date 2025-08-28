// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

import { AnyAction } from 'redux';
import { ProjectOrTaskOrJob } from 'cvat-core-wrapper';
import { TrainingActionTypes } from 'actions/training-actions';

export interface TrainingState {
    instance: ProjectOrTaskOrJob | null;
    jobId: string | null;
    exportUrl: string | null;
}

export const defaultState: TrainingState = {
    instance: null,
    jobId: null,
    exportUrl: null,
};

export default function (state: TrainingState = defaultState, action: AnyAction): TrainingState {
    switch (action.type) {
        case TrainingActionTypes.OPEN_TRAIN_DATASET_MODAL: {
            return { ...state, instance: action.payload.instance, exportUrl: null };
        }
        case TrainingActionTypes.CLOSE_TRAIN_DATASET_MODAL: {
            return { ...state, instance: null, exportUrl: null };
        }
        case TrainingActionTypes.SET_CURRENT_JOB_ID: {
            return { ...state, jobId: action.payload.jobId };
        }
        case TrainingActionTypes.OPEN_TRAIN_DATASET_FROM_URL_MODAL: {
            return { ...state, exportUrl: action.payload.exportUrl, instance: null };
        }
        default:
            return state;
    }
}
