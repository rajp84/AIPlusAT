// Copyright (C) CVAT.ai Corporation
// SPDX-License-Identifier: MIT

import React, { useCallback, useState } from 'react';
import { connect, useDispatch } from 'react-redux';
import Modal from 'antd/lib/modal';
import Form from 'antd/lib/form';
import InputNumber from 'antd/lib/input-number';
import Switch from 'antd/lib/switch';
import Text from 'antd/lib/typography/Text';
import Input from 'antd/lib/input';
import Space from 'antd/lib/space';
import notification from 'antd/lib/notification';
import { requestsActions } from 'actions/requests-actions';

import { CombinedState } from 'reducers';
import { Project, Task, Job } from 'cvat-core-wrapper';
import { trainingActions, startTrainAsRequestFromExportAsync, startTrainAsRequestFromExportUrlAsync } from 'actions/training-actions';

type FormValues = {
    imgsz: number;
    epochs: number;
    batch: number;
    device: string;
    workers: number;
    model: string;
    trt_half: boolean;
};

const initialValues: FormValues = {
    imgsz: 512,
    epochs: 100,
    batch: 16,
    device: '0',
    workers: 8,
    model: 'yolov8s.pt',
    trt_half: true,
};

function TrainDatasetModal(props: Readonly<StateToProps>): JSX.Element {
    const { instance, exportUrl } = props;
    const [form] = Form.useForm<FormValues>();
    const dispatch = useDispatch();
    const [submitting, setSubmitting] = useState(false);

    const close = useCallback(() => {
        dispatch(trainingActions.closeTrainDatasetModal());
    }, []);

    const onFinish = useCallback(async (values: FormValues) => {
        setSubmitting(true);
        if (instance) {
            const { rqId, jobId } = await (dispatch(startTrainAsRequestFromExportAsync(instance, values) as any));
            notification.success({
                message: 'Training request enqueued',
                description: rqId ? `Request ID: ${rqId}. See Requests page for progress.` : 'See Requests page for progress.',
                duration: 4,
            });
            if (jobId) dispatch(trainingActions.setCurrentJobId(jobId));
            dispatch(requestsActions.getRequests({}, true));
            close();
            setSubmitting(false);
            return;
        }
        if (exportUrl) {
            const { rqId, jobId } = await (dispatch(startTrainAsRequestFromExportUrlAsync(exportUrl, values) as any));
            notification.success({
                message: 'Training request enqueued',
                description: rqId ? `Request ID: ${rqId}. See Requests page for progress.` : 'See Requests page for progress.',
                duration: 4,
            });
            if (jobId) dispatch(trainingActions.setCurrentJobId(jobId));
            dispatch(requestsActions.getRequests({}, true));
            close();
        }
        setSubmitting(false);
    }, [instance, exportUrl]);

    return (
        <Modal
            title={<Text strong>Train model from dataset</Text>}
            className='cvat-modal-train-dataset'
            open={!!instance || !!exportUrl}
            onCancel={close}
            onOk={() => form.submit()}
            destroyOnClose
        >
            <Form<FormValues>
                form={form}
                layout='vertical'
                initialValues={initialValues}
                onFinish={onFinish}
            >
                <Form.Item name='model' label={<Text strong>Base model</Text>}>
                    <Input />
                </Form.Item>
                <Space size='large'>
                    <Form.Item name='imgsz' label={<Text strong>Image size</Text>}>
                        <InputNumber min={256} max={2048} step={64} />
                    </Form.Item>
                    <Form.Item name='epochs' label={<Text strong>Epochs</Text>}>
                        <InputNumber min={1} max={2000} />
                    </Form.Item>
                    <Form.Item name='batch' label={<Text strong>Batch</Text>}>
                        <InputNumber min={1} max={512} />
                    </Form.Item>
                </Space>
                <Space size='large'>
                    <Form.Item name='device' label={<Text strong>Device</Text>}>
                        <Input />
                    </Form.Item>
                    <Form.Item name='workers' label={<Text strong>Workers</Text>}>
                        <InputNumber min={0} max={32} />
                    </Form.Item>
                </Space>
                <Form.Item name='trt_half' label={<Text strong>TensorRT FP16</Text>} valuePropName='checked'>
                    <Switch />
                </Form.Item>
            </Form>
            {submitting && (
                <div style={{ display: 'flex', justifyContent: 'center', paddingTop: 8 }}>
                    <span className='cvat-spinner' />
                </div>
            )}
        </Modal>
    );
}

interface StateToProps {
    instance: Project | Task | Job | null;
    exportUrl: string | null;
}

function mapStateToProps(state: CombinedState): StateToProps {
    return { instance: state.training.instance as any, exportUrl: state.training.exportUrl };
}

export default connect(mapStateToProps)(TrainDatasetModal);
