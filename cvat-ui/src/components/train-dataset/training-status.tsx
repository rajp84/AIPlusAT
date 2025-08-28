import React, { useEffect, useRef, useState } from 'react';
import { connect, useDispatch } from 'react-redux';
import Modal from 'antd/lib/modal';
import Text from 'antd/lib/typography/Text';
import Button from 'antd/lib/button';
import notification from 'antd/lib/notification';
import { CombinedState } from 'reducers';
import { trainingActions } from 'actions/training-actions';

function TrainingStatus(props: Readonly<StateToProps>): JSX.Element | null {
    const { jobId } = props;
    const dispatch = useDispatch();
    const [logs, setLogs] = useState('');
    const [status, setStatus] = useState('');
    const notifiedRef = useRef(false);
    const preRef = useRef<HTMLPreElement | null>(null);
    const offsetRef = useRef(0);
    const baseRef = useRef<string | null>(null);

    useEffect(() => {
        if (!jobId) return () => {};
        localStorage.setItem('trainer.currentJobId', jobId);
        let base = (process.env.NUCLIO_TRAINER_URL || '').replace(/\/$/, '');
        let mounted = true;
        offsetRef.current = 0;
        baseRef.current = null;
        const tick = async (): Promise<void> => {
            try {
                if (!base) {
                    try {
                        const core = (await import('cvat-core-wrapper')).getCore();
                        const functions = await core.lambda.list();
                        const trainer = functions.models.find((f: any) => String(f.name || '').toLowerCase().includes('trainer')) ||
                            functions.models.find((f: any) => String(f.name || '').toLowerCase().includes('aiplus') &&
                                String(f.name || '').toLowerCase().includes('train')) ||
                            functions.models[0];
                        if (trainer?.url) base = String(trainer.url).replace(/\/$/, '');
                    } catch { /* ignore */ }
                }
                if (!base) throw new Error('No trainer URL');
                baseRef.current = base;
                const s = await fetch(`${base}/status?id=${encodeURIComponent(jobId)}`).then((r) => r.json());
                // long-poll chunk
                const url = `${base}/logs/stream?id=${encodeURIComponent(jobId)}&offset=${offsetRef.current}`;
                const chunkResp = await fetch(url);
                if (chunkResp.ok) {
                    const ct = chunkResp.headers.get('content-type') || '';
                    if (ct.includes('application/json')) {
                        const { chunk, offset: newOffset } = await chunkResp.json();
                        if (mounted && chunk) {
                            setLogs((prev: string): string => {
                            // normalize to newline-terminated chunks
                                const appended = chunk.endsWith('\n') ? chunk : `${chunk}\n`;
                                const combined = prev + appended;
                                const lines = combined.split(/\r?\n/);
                                const limited = lines.slice(Math.max(0, lines.length - 200));
                                return limited.join('\n');
                            });
                        }
                        offsetRef.current = newOffset || offsetRef.current;
                    } else {
                        const txt = await chunkResp.text();
                        if (mounted && txt) {
                            setLogs((prev: string): string => {
                                const appended = txt.endsWith('\n') ? txt : `${txt}\n`;
                                const combined = prev + appended;
                                const lines = combined.split(/\r?\n/);
                                const limited = lines.slice(Math.max(0, lines.length - 200));
                                return limited.join('\n');
                            });
                        }
                    }
                } else {
                    // Fallback to full logs if stream is not supported
                    const full = await fetch(`${base}/logs?id=${encodeURIComponent(jobId)}`).then((r) => r.text());
                    if (mounted) {
                        setLogs((): string => {
                            const lines = full.split(/\r?\n/);
                            const limited = lines.slice(Math.max(0, lines.length - 200));
                            return limited.join('\n');
                        });
                    }
                }
                if (mounted) {
                    setStatus(s.status || '');
                    if ((s.status === 'finished' || s.status === 'failed') && !notifiedRef.current) {
                        notifiedRef.current = true;
                        notification.info({
                            message: s.status === 'finished' ? 'Training complete' : 'Training failed',
                            description: s.status === 'finished' ?
                                'Your training job has finished successfully.' :
                                'Your training job failed. Check logs.',
                            duration: 5,
                        });
                    }
                    if (s.status === 'finished' || s.status === 'failed') {
                        return;
                    }
                }
            } catch { /* ignore */ }
            if (mounted) setTimeout(tick, 1000);
        };
        tick();
        return () => { mounted = false; };
    }, [jobId]);

    // Auto-scroll to bottom on new logs
    useEffect((): void => {
        if (preRef.current) {
            preRef.current.scrollTop = preRef.current.scrollHeight;
        }
    }, [logs]);

    if (!jobId) return null;

    return (
        <Modal
            title={(
                <Text strong>
Training status (
                    {status || 'running'}
)
                </Text>
            )}
            open={!!jobId}
            onCancel={() => {
                dispatch(trainingActions.setCurrentJobId(null));
                localStorage.removeItem('trainer.currentJobId');
            }}
            footer={(
                <Button onClick={() => {
                    dispatch(trainingActions.setCurrentJobId(null));
                    localStorage.removeItem('trainer.currentJobId');
                }}
                >
                    Close
                </Button>
            )}
            className='cvat-modal-training-status'
            width={900}
        >
            <pre
                ref={preRef}
                style={{ maxHeight: 480, overflow: 'auto', whiteSpace: 'pre-wrap' }}
            >
                {logs}
            </pre>
        </Modal>
    );
}

interface StateToProps { jobId: string | null }
function mapStateToProps(state: CombinedState): StateToProps {
    // Restore last job id if modal is reopened later
    const current = state.training.jobId || localStorage.getItem('trainer.currentJobId') || null;
    return { jobId: current };
}

export default connect(mapStateToProps)(TrainingStatus);
