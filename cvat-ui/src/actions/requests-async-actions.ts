// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { ThunkAction } from 'utils/redux';
import { CombinedState, RequestsQuery } from 'reducers';
import {
    getCore, RQStatus, Request, Project, Task, Job, StorageLocation,
} from 'cvat-core-wrapper';
import { listenExportBackupAsync, listenExportDatasetAsync } from './export-actions';
import {
    RequestInstanceType, listen, requestsActions,
} from './requests-actions';
import { listenImportBackupAsync, listenImportDatasetAsync } from './import-actions';

const core = getCore();

export interface RequestParams {
    id: string;
    type: string;
    instance?: Project | Task | Job;
    location?: StorageLocation;
}

export function getRequestsAsync(query: Partial<RequestsQuery> = {}): ThunkAction {
    return async (dispatch, getState): Promise<void> => {
        dispatch(requestsActions.getRequests(query));

        const state: CombinedState = getState();

        try {
            const requests = await core.requests.list();
            dispatch(requestsActions.getRequestsSuccess(requests));

            requests
                .filter((request: Request) => [RQStatus.STARTED, RQStatus.QUEUED].includes(request.status))
                .forEach((request: Request): void => {
                    const {
                        id: rqID,
                        status,
                        operation: {
                            type, target, format, taskID, projectID, jobID,
                        },
                    } = request;

                    const isRequestFinished = [RQStatus.FINISHED, RQStatus.FAILED].includes(status);
                    if (state.requests.requests[rqID] || isRequestFinished) {
                        return;
                    }

                    let instance: RequestInstanceType | null = null;

                    const [operationType, operationTarget] = type.split(':');
                    if (target === 'task') {
                        instance = { id: taskID as number, type: target };
                    } else if (target === 'job') {
                        instance = { id: jobID as number, type: target };
                    } else if (target === 'project') {
                        instance = { id: projectID as number, type: target };
                    }

                    if (operationType === 'lambda') {
                        // For lambda we don't have /api/requests/<id> polling; re-fetch list in the background
                        setTimeout(() => dispatch(requestsActions.getRequests({ ...query }, false)), 2000);
                    } else if (operationType === 'export') {
                        if (operationTarget === 'backup') {
                            listenExportBackupAsync(rqID, dispatch, { instance: instance as RequestInstanceType });
                        } else if (operationTarget === 'dataset' || operationTarget === 'annotations') {
                            const safeFormat: string = (format || '') as string;
                            listenExportDatasetAsync(
                                rqID,
                                dispatch,
                                { instance: instance as RequestInstanceType, format: safeFormat, saveImages: type.includes('dataset') },
                            );
                        }
                    } else if (operationType === 'import') {
                        if (operationTarget === 'backup') {
                            listenImportBackupAsync(rqID, dispatch, { instanceType: (instance as RequestInstanceType).type as 'project' | 'task' });
                        } else if (operationTarget === 'dataset' || operationTarget === 'annotations') {
                            listenImportDatasetAsync(
                                rqID,
                                dispatch,
                                { instance: instance as RequestInstanceType },
                            );
                        }
                    } else if (operationType === 'create') {
                        if (operationTarget === 'task') {
                            listen(rqID, dispatch);
                        }
                    }
                });
        } catch (error) {
            dispatch(requestsActions.getRequestsFailed(error));
        }
    };
}

export function cancelRequestAsync(request: Request): ThunkAction {
    return async (dispatch): Promise<void> => {
        dispatch(requestsActions.cancelRequest());

        try {
            await core.requests.cancel(request.id);
            dispatch(requestsActions.cancelRequestSuccess(request));
        } catch (error) {
            dispatch(requestsActions.cancelRequestFailed(request, error));
        }
    };
}

export function terminateLambdaRequestAsync(request: Request): ThunkAction {
    return async (dispatch): Promise<void> => {
        try {
            // DELETE /api/lambda/requests/<rq_id> with CSRF
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
            const encoded = encodeURIComponent(request.id);
            const resp = await fetch(`/api/lambda/requests/${encoded}`, {
                method: 'DELETE',
                headers: { 'X-CSRFToken': csrf, 'X-Requested-With': 'XMLHttpRequest' },
                credentials: 'include',
            });
            if (!resp.ok) {
                const txt = await resp.text();
                throw new Error(`Failed to terminate: ${resp.status} ${txt}`);
            }
            // Refresh list after termination
            const requests = await core.requests.list();
            dispatch(requestsActions.getRequestsSuccess(requests));
        } catch (error) {
            dispatch(requestsActions.getRequestsFailed(error));
        }
    };
}
