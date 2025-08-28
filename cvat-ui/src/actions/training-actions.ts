// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

import { ActionUnion, createAction, ThunkAction } from 'utils/redux';
import {
    ProjectOrTaskOrJob, Storage, StorageLocation, getCore,
} from 'cvat-core-wrapper';

export enum TrainingActionTypes {
    OPEN_TRAIN_DATASET_MODAL = 'OPEN_TRAIN_DATASET_MODAL',
    CLOSE_TRAIN_DATASET_MODAL = 'CLOSE_TRAIN_DATASET_MODAL',
    SET_CURRENT_JOB_ID = 'SET_CURRENT_JOB_ID',
    OPEN_TRAIN_DATASET_FROM_URL_MODAL = 'OPEN_TRAIN_DATASET_FROM_URL_MODAL',
}

const core = getCore();

export const trainingActions = {
    openTrainDatasetModal: (instance: ProjectOrTaskOrJob) => (
        createAction(TrainingActionTypes.OPEN_TRAIN_DATASET_MODAL, { instance })
    ),
    closeTrainDatasetModal: () => (
        createAction(TrainingActionTypes.CLOSE_TRAIN_DATASET_MODAL)
    ),
    setCurrentJobId: (jobId: string | null) => (
        createAction(TrainingActionTypes.SET_CURRENT_JOB_ID, { jobId })
    ),
    openTrainDatasetFromUrlModal: (exportUrl: string) => (
        createAction(TrainingActionTypes.OPEN_TRAIN_DATASET_FROM_URL_MODAL, { exportUrl })
    ),
};

export type TrainingActions = ActionUnion<typeof trainingActions>;

export interface TrainParams {
    imgsz?: number;
    epochs?: number;
    batch?: number;
    device?: string;
    workers?: number;
    model?: string;
    trt_half?: boolean;
    build_with_python_trt?: boolean;
}

export const startTrainFromExportAsync = (
    instance: ProjectOrTaskOrJob,
    params: TrainParams,
): ThunkAction<Promise<{ jobId: string }>> => async (dispatch) => {
    // 1) Export dataset as CVAT YOLO 1.1 (with images)
    const exportName = `train_${instance.id}_${Date.now()}.zip`;
    const rqID = await instance.annotations.exportDataset(
        'YOLO 1.1',
        true,
        true,
        new Storage({ location: StorageLocation.LOCAL }),
        exportName,
    );

    if (!rqID) {
        throw new Error('Failed to start dataset export');
    }

    const result = await core.requests.listen(rqID, { callback: () => {} });
    const url = result?.url as string | undefined;
    if (!url) {
        throw new Error('Export finished without a downloadable URL');
    }

    // 2) Trigger Nuclio trainer
    const trainerURL = (process.env.NUCLIO_TRAINER_URL || '').replace(/\/$/, '');
    if (!trainerURL) {
        throw new Error('NUCLIO_TRAINER_URL is not configured');
    }

    const resp = await fetch(`${trainerURL}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            cvat_export_url: url,
            imgsz: params.imgsz ?? 512,
            epochs: params.epochs ?? 100,
            batch: params.batch ?? 16,
            device: params.device ?? '0',
            workers: params.workers ?? 8,
            model: params.model ?? 'yolov8s.pt',
            build_with_python_trt: true,
            trt_half: params.trt_half ?? true,
        }),
    });

    if (!resp.ok) {
        let detail = '';
        try {
            const t = await resp.text();
            try {
                const j = JSON.parse(t);
                detail = j.error || t;
            } catch {
                detail = t;
            }
        } catch { /* ignore */ }
        throw new Error(`Trainer responded with ${resp.status}${detail ? `: ${detail}` : ''}`);
    }
    const data = await resp.json();
    const jobId = data.job_id as string;
    dispatch(trainingActions.setCurrentJobId(jobId));
    return { jobId };
};

export const startTrainFromExportUrlAsync = (
    exportUrl: string,
    params: TrainParams,
): ThunkAction<Promise<{ jobId: string }>> => async (dispatch) => {
    // Discover trainer via CVAT lambda functions
    const coreInstance = getCore();
    const functions = await coreInstance.lambda.list();
    const trainer = functions.models.find((f: any) => String(f.name || '').toLowerCase().includes('trainer')) ||
        functions.models.find((f: any) => String(f.name || '').toLowerCase().includes('aiplus') && String(f.name || '').toLowerCase().includes('train')) ||
        functions.models[0];

    if (!trainer) throw new Error('No serverless functions found to route training');

    // Prefer discovered direct url; fallback to env var; if neither, error early
    const directUrl: string = (trainer?.url as any) || (process.env.NUCLIO_TRAINER_URL || '');
    const base = (directUrl || '').replace(/\/$/, '');
    const url = base ? `${base}/train` : '';

    if (!url) throw new Error('Trainer URL not discovered. Ensure Nuclio function is running and CVAT lists a port.');

    // Always ask trainer to download inside the container (credentials/token can be provided on trainer env)
    const resp = await fetch(`${base}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            cvat_export_url: exportUrl,
            imgsz: params.imgsz ?? 512,
            epochs: params.epochs ?? 100,
            batch: params.batch ?? 16,
            device: params.device ?? '0',
            workers: params.workers ?? 8,
            model: params.model ?? 'yolov8s.pt',
            build_with_python_trt: true,
            trt_half: params.trt_half ?? true,
            __action: 'train',
        }),
    });
    if (!resp.ok) {
        let detail = '';
        try {
            const t = await resp.text();
            try {
                const j = JSON.parse(t);
                detail = j.error || t;
            } catch {
                detail = t;
            }
        } catch { /* ignore */ }
        throw new Error(`Trainer responded with ${resp.status}${detail ? `: ${detail}` : ''}`);
    }
    const data = await resp.json();
    const jobId = data.job_id as string;
    dispatch(trainingActions.setCurrentJobId(jobId));
    return { jobId };
};

// Request-based training: enqueue a CVAT RQ job so it appears on the Requests page,
// then poll the request until the Nuclio job id is available, and open the live logs.
export const startTrainAsRequestFromExportAsync = (
    instance: ProjectOrTaskOrJob,
    params: TrainParams,
): ThunkAction<Promise<{ rqId: string; jobId: string | null }>> => async (dispatch) => {
    // Step 1: Export dataset to get a download URL
    const exportName = `train_${(instance as any).id}_${Date.now()}.zip`;
    const rqID = await instance.annotations.exportDataset(
        'YOLO 1.1',
        true,
        true,
        new Storage({ location: StorageLocation.LOCAL }),
        exportName,
    );
    if (!rqID) throw new Error('Failed to start dataset export');
    const result = await core.requests.listen(rqID, { callback: () => {} });
    const exportUrl = result?.url as string | undefined;
    if (!exportUrl) throw new Error('Export finished without a downloadable URL');

    // Step 2: Find trainer function id
    const functions = await core.lambda.list();
    const trainer = functions.models.find((f: any) => String(f.name || '').toLowerCase().includes('trainer')) ||
        functions.models.find((f: any) => String(f.name || '').toLowerCase().includes('aiplus') && String(f.name || '').toLowerCase().includes('train')) ||
        functions.models[0];
    if (!trainer) throw new Error('No serverless functions found to route training');
    const functionId: string = (trainer?.id as any) || (trainer?.name as any);

    // Step 3: Enqueue CVAT request for training
    const body = {
        function: functionId,
        task: (instance as any).id,
        params: {
            __action: 'train',
            cvat_export_url: exportUrl,
            imgsz: params.imgsz ?? 512,
            epochs: params.epochs ?? 100,
            batch: params.batch ?? 16,
            device: params.device ?? '0',
            workers: params.workers ?? 8,
            model: params.model ?? 'yolov8s.pt',
            build_with_python_trt: true,
            trt_half: params.trt_half ?? true,
        },
    } as any;
    // CSRF
    const getCookie = (name: string): string => {
        const raw = document.cookie || '';
        if (!raw) return '';
        const pairs = raw.split('; ');
        for (const pair of pairs) {
            const eqIdx = pair.indexOf('=');
            const key = eqIdx === -1 ? pair : pair.slice(0, eqIdx);
            if (decodeURIComponent(key) === name) {
                const val = eqIdx === -1 ? '' : pair.slice(eqIdx + 1);
                return decodeURIComponent(val);
            }
        }
        return '';
    };
    const csrf = getCookie('csrftoken') || getCookie('cvatcsrftoken');
    const resp = await fetch('/api/lambda/requests', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf, 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'include',
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const t = await resp.text();
        throw new Error(`Failed to enqueue training request: ${resp.status} ${t}`);
    }
    const rqData = await resp.json();
    const rqId: string = rqData.id;
    localStorage.setItem('trainer.rqId', rqId);

    // Step 4: Poll request details until nuclio_job_id appears, then open modal
    let jobId: string | null = null;
    const maxMs = 15000; // 15s
    const start = Date.now();
    while (Date.now() - start < maxMs) {
        try {
            const det = await fetch(`/api/lambda/requests/${encodeURIComponent(rqId)}`).then((r) => r.json());
            if (det && det.nuclio_job_id) {
                jobId = String(det.nuclio_job_id);
                break;
            }
        } catch { /* ignore */ }
        await new Promise<void>((r) => { setTimeout(() => { r(); }, 1000); });
    }
    if (jobId) {
        dispatch(trainingActions.setCurrentJobId(jobId));
    }
    return { rqId, jobId };
};

export const startTrainAsRequestFromExportUrlAsync = (
    exportUrl: string,
    params: TrainParams,
): ThunkAction<Promise<{ rqId: string; jobId: string | null }>> => async (dispatch) => {
    // Step 1: Find trainer function id
    const functions = await core.lambda.list();
    const trainer = functions.models.find((f: any) => String(f.name || '').toLowerCase().includes('trainer')) ||
        functions.models.find((f: any) => String(f.name || '').toLowerCase().includes('aiplus') && String(f.name || '').toLowerCase().includes('train')) ||
        functions.models[0];
    if (!trainer) throw new Error('No serverless functions found to route training');
    const functionId: string = (trainer?.id as any) || (trainer?.name as any);

    // Derive task id from export URL if possible: /api/tasks/<id>/dataset/download
    let taskIdFromUrl: number | undefined;
    try {
        const m = exportUrl.match(/\/tasks\/(\d+)\//);
        if (m && m[1]) taskIdFromUrl = parseInt(m[1], 10);
    } catch { /* ignore */ }

    // Step 2: Enqueue CVAT request for training
    const body = {
        function: functionId,
        task: taskIdFromUrl,
        params: {
            __action: 'train',
            cvat_export_url: exportUrl,
            imgsz: params.imgsz ?? 512,
            epochs: params.epochs ?? 100,
            batch: params.batch ?? 16,
            device: params.device ?? '0',
            workers: params.workers ?? 8,
            model: params.model ?? 'yolov8s.pt',
            build_with_python_trt: true,
            trt_half: params.trt_half ?? true,
        },
    } as any;
    const getCookie = (name: string): string => {
        const raw = document.cookie || '';
        if (!raw) return '';
        const pairs = raw.split('; ');
        for (const pair of pairs) {
            const eqIdx = pair.indexOf('=');
            const key = eqIdx === -1 ? pair : pair.slice(0, eqIdx);
            if (decodeURIComponent(key) === name) {
                const val = eqIdx === -1 ? '' : pair.slice(eqIdx + 1);
                return decodeURIComponent(val);
            }
        }
        return '';
    };
    const csrf = getCookie('csrftoken') || getCookie('cvatcsrftoken');
    const resp = await fetch('/api/lambda/requests', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf, 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'include',
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const t = await resp.text();
        throw new Error(`Failed to enqueue training request: ${resp.status} ${t}`);
    }
    const rqData = await resp.json();
    const rqId: string = rqData.id;
    localStorage.setItem('trainer.rqId', rqId);

    // Step 3: Poll request details until nuclio_job_id appears
    let jobId: string | null = null;
    const maxMs = 15000;
    const start = Date.now();
    while (Date.now() - start < maxMs) {
        try {
            const det = await fetch(`/api/lambda/requests/${encodeURIComponent(rqId)}`).then((r) => r.json());
            if (det && det.nuclio_job_id) {
                jobId = String(det.nuclio_job_id);
                break;
            }
        } catch { /* ignore */ }
        await new Promise<void>((r) => { setTimeout(() => { r(); }, 1000); });
    }
    if (jobId) {
        dispatch(trainingActions.setCurrentJobId(jobId));
    }
    return { rqId, jobId };
};

export const resumeLastTrainingAsync = (): ThunkAction<Promise<string | null>> => async (dispatch) => {
    // Discover trainer via CVAT lambda functions
    const coreInstance = getCore();
    const functions = await coreInstance.lambda.list();
    const trainer = functions.models.find((f: any) => String(f.name || '').toLowerCase().includes('trainer')) ||
        functions.models.find((f: any) => String(f.name || '').toLowerCase().includes('aiplus') && String(f.name || '').toLowerCase().includes('train')) ||
        functions.models[0];
    const directUrl: string = (trainer?.url as any) || (process.env.NUCLIO_TRAINER_URL || '');
    const base = (directUrl || '').replace(/\/$/, '');
    if (!base) return null;

    try {
        const resp = await fetch(`${base}/jobs/latest`);
        if (!resp.ok) return null;
        const data = await resp.json();
        const jobId: string | undefined = data?.job_id;
        if (!jobId) return null;
        dispatch(trainingActions.setCurrentJobId(jobId));
        return jobId;
    } catch {
        return null;
    }
};
