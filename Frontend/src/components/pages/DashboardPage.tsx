import React, { useState, useCallback, useEffect } from 'react';
import { Upload, Loader2, Grid, User, Settings, LogOut, ChevronDown, Trash2 } from 'lucide-react';
import type { DashboardPageProps, AnalysisResult } from '../../types';
import { API_BASE_URL } from '../../types';
import AlertMessage from '../ui/AlertMessage';
import { IconMicroscope, IconLeaf } from '../ui/Icons';

/* ========================================================================
   PROFILE DROPDOWN
======================================================================== */
interface ProfileDropdownProps {
    userEmail: string;
    onLogout: () => void;
    userId: string | null;
}

const ProfileDropdown: React.FC<ProfileDropdownProps> = ({ userEmail, onLogout }) => {
    const [isOpen, setIsOpen] = useState(false);

    return (
        <div className="relative z-50 pt-8">
            <button
                aria-label="Toggle profile dropdown"
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center space-x-2 bg-gray-700 hover:bg-gray-800 text-amber-300 p-2 rounded-full transition-colors shadow-md"
            >
                <div className="w-8 h-8 rounded-full bg-amber-500 flex items-center justify-center text-gray-900 font-bold text-sm">
                    {userEmail[0]?.toUpperCase()}
                </div>
                <span className="hidden sm:inline font-semibold">{userEmail}</span>
                <ChevronDown className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>

            {isOpen && (
                <div className="absolute right-0 mt-2 w-64 bg-gray-800 rounded-xl shadow-2xl border border-gray-700 py-2 text-amber-300">
                    <div className="px-4 py-2 text-sm font-semibold border-b border-gray-700">Profile</div>

                    <button aria-label="My Account" className="flex items-center space-x-3 w-full px-4 py-2 hover:bg-gray-700 transition-colors">
                        <User className="h-4 w-4 text-amber-400" /> My Account
                    </button>

                    <button aria-label="Update Settings" className="flex items-center space-x-3 w-full px-4 py-2 hover:bg-gray-700 transition-colors">
                        <Settings className="h-4 w-4 text-amber-400" /> Update Settings
                    </button>

                    <button
                        aria-label="Logout"
                        onClick={onLogout}
                        className="flex items-center space-x-3 w-full px-4 py-2 hover:bg-red-600 transition-colors text-red-500 border-t mt-1"
                    >
                        <LogOut className="h-4 w-4" /> Logout
                    </button>
                </div>
            )}
        </div>
    );
};

/* ========================================================================
   DASHBOARD HEADER
======================================================================== */
const DashboardHeader: React.FC<DashboardPageProps> = (props) => (
    <div className="fixed top-0 left-0 right-0 bg-gray-900/95 backdrop-blur-md py-4 px-5 shadow-xl z-50 border-b border-gray-800">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
            <div className="text-amber-400 text-xl sm:text-2xl font-extrabold flex items-center space-x-2">
                <IconLeaf className="text-amber-500 h-6 w-6" />
                <span>AgroScan AI | Dashboard</span>
            </div>
            <ProfileDropdown {...props} />
        </div>
    </div>
);

/* ========================================================================
   MAIN DASHBOARD PAGE
======================================================================== */
const DashboardPage: React.FC<DashboardPageProps> = ({ userToken, userId, userEmail, onLogout }) => {
    useEffect(() => {
        const originalMargin = document.body.style.margin;
        const originalPadding = document.body.style.padding;
        document.body.style.margin = '0';
        document.body.style.padding = '0';
        return () => {
            document.body.style.margin = originalMargin;
            document.body.style.padding = originalPadding;
        };
    }, []);

    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [results, setResults] = useState<AnalysisResult[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [uploadMessage, setUploadMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    /* =====================================================================
       DELETE SCAN
    ===================================================================== */
    const handleDeleteScan = async (scanId: string | number | undefined) => {
        if (!scanId) return;

        try {
            const res = await fetch(`${API_BASE_URL}/delete_scan/${scanId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    ...(userToken ? { Authorization: `Bearer ${userToken}` } : {})
                }
            });

            await res.json().catch(() => null);

            if (!res.ok) return setUploadMessage({ text: "Failed to delete scan.", type: 'error' });

            setResults(prev => prev.filter(r => r.scan_id !== scanId));
            setUploadMessage({ text: "Scan deleted successfully.", type: 'success' });
        } catch {
            setUploadMessage({ text: "Network error while deleting scan.", type: 'error' });
        }
    };

    /* =====================================================================
       FETCH SAVED SCANS
    ===================================================================== */
    const fetchSavedScans = useCallback(async () => {
        if (!userEmail) return setResults([]);

        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE_URL}/get_scans/${encodeURIComponent(userEmail)}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...(userToken ? { Authorization: `Bearer ${userToken}` } : {})
                }
            });

            const json = await res.json().catch(() => null);
            if (!res.ok || !json?.scans) return setResults([]);

            setResults(json.scans.map((s: any) => ({
                filename: `Scan ID: ${s.scan_id}`,
                prediction: s.prediction || 'Unknown',
                confidence: s.confidence ?? 0,
                timestamp: s.date ? new Date(s.date).toLocaleString() : 'Date Unavailable',
                recommendation: s.treatment_recommendation || 'No recommendation.',
                status: s.status,
                message: s.message,
                image: s.image_link,
                scan_id: s.scan_id
            })));
        } finally {
            setIsLoading(false);
        }
    }, [userEmail, userToken]);

    useEffect(() => {
        if (userEmail) fetchSavedScans();
    }, [userEmail, fetchSavedScans]);

    /* =====================================================================
       FILE UPLOAD
    ===================================================================== */
    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files?.length) {
            const file = event.target.files[0];

            if (file.size > 5 * 1024 * 1024) {
                setUploadMessage({ text: 'File too large. Max 5MB.', type: 'error' });
                return;
            }

            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setUploadMessage(null);
        }
    };

    /* =====================================================================
       UPLOAD AND PREDICT
    ===================================================================== */
    const handleUpload = useCallback(async (e: React.FormEvent) => {
        e.preventDefault();
        if (!selectedFile)
            return setUploadMessage({ text: 'Please select an image.', type: 'error' });

        setIsLoading(true);
        setUploadMessage(null);

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('user_email', userEmail);

            const res = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { ...(userToken ? { Authorization: `Bearer ${userToken}` } : {}) },
                body: formData
            });

            const data = await res.json().catch(() => null);

            if (!res.ok) {
                setUploadMessage({ text: data?.message || 'Analysis failed.', type: 'error' });
                return;
            }

            const newResult: AnalysisResult = {
                filename: selectedFile.name,
                prediction: data.prediction,
                confidence: data.confidence,
                timestamp: new Date().toLocaleString(),
                status: data.status,
                message: data.message,
                recommendation: data.recommendation,
                scan_id: data.save_status === 'SAVED_SUCCESS' ? data.scan_id : undefined
            };

            if (data.save_status === 'SAVED_SUCCESS') {
                await fetchSavedScans();
                setUploadMessage({ text: 'Scan saved to history.', type: 'success' });
            }

            setResults(prev => [newResult, ...prev]);
            setSelectedFile(null);
            setPreviewUrl(null);

        } catch {
            setUploadMessage({ text: 'Network error.', type: 'error' });

        } finally {
            setIsLoading(false);
        }
    }, [selectedFile, userToken, userEmail, fetchSavedScans]);

    /* =====================================================================
       PAGE UI
    ===================================================================== */
    return (
        <>
            <DashboardHeader userEmail={userEmail} userId={userId} userToken={userToken} onLogout={onLogout} />

            <div className="min-h-screen bg-gray-900 pt-[4.5rem] p-4 sm:p-8 text-amber-300">
                <div className="max-w-7xl mx-auto">

                    <header className="mb-8 p-6 bg-gray-800 rounded-2xl shadow-xl border-l-4 border-amber-500">
                        <h1 className="text-2xl sm:text-3xl font-extrabold flex items-center space-x-3">
                            <Grid className="h-6 w-6 text-amber-500" />
                            <span>Farmer Dashboard</span>
                        </h1>
                        <p className="mt-2 text-amber-400">
                            Welcome, <span className="font-mono p-1 bg-gray-700 rounded">{userEmail}</span>.
                        </p>
                    </header>

                    <div className="grid lg:grid-cols-3 gap-8">

                        {/* ======================= Upload Section ======================= */}
                        <div className="lg:col-span-1 bg-gray-800 p-6 rounded-2xl shadow-xl h-fit sticky top-[90px]">
                            <h2 className="text-xl sm:text-2xl font-bold mb-4 border-b border-amber-500 pb-2">New Scan</h2>

                            <AlertMessage message={uploadMessage?.text || null} type={uploadMessage?.type || null} />

                            <form onSubmit={handleUpload}>
                                <div className="mb-4">
                                    <label className="block text-sm font-medium mb-2">Select Tea Leaf Image (JPG/PNG)</label>

                                    <label
                                        htmlFor="file-upload"
                                        className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-amber-500 rounded-xl cursor-pointer bg-gray-900 hover:bg-gray-800 transition-colors"
                                    >
                                        {previewUrl ? (
                                            <img src={previewUrl} alt={`Preview of ${selectedFile?.name}`} className="h-full w-full object-cover rounded-xl p-1" />
                                        ) : (
                                            <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                                <Upload className="w-10 h-10 mb-3 text-amber-500" />
                                                <p className="text-sm"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                                                <p className="text-xs">{selectedFile?.name || 'Max 5MB'}</p>
                                            </div>
                                        )}

                                        <input
                                            id="file-upload"
                                            type="file"
                                            className="hidden"
                                            accept="image/jpeg,image/png"
                                            onChange={handleFileChange}
                                            aria-label="Select tea leaf image"
                                        />
                                    </label>
                                </div>

                                <button
                                    type="submit"
                                    disabled={isLoading || !selectedFile}
                                    className={`w-full py-3 rounded-xl font-semibold flex items-center justify-center space-x-2 transition-all ${
                                        isLoading || !selectedFile
                                            ? 'bg-amber-700/50 cursor-not-allowed'
                                            : 'bg-amber-500 hover:bg-amber-400'
                                    }`}
                                    aria-label="Run AI Scan"
                                >
                                    {isLoading ? (
                                        <>
                                            <Loader2 className="animate-spin h-5 w-5" />
                                            <span>Analyzing...</span>
                                        </>
                                    ) : (
                                        <>
                                            <Upload className="h-5 w-5" />
                                            <span>Run AI Scan</span>
                                        </>
                                    )}
                                </button>
                            </form>
                        </div>

                        {/* ======================= Scan History ======================= */}
                        <div className="lg:col-span-2 bg-gray-800 p-6 rounded-2xl shadow-xl">
                            <h2 className="text-xl sm:text-2xl font-bold mb-6 border-b border-amber-500 pb-2">
                                Scan History ({results.length})
                            </h2>

                            {results.length === 0 ? (
                                <div className="text-center py-12 bg-gray-900 rounded-xl text-amber-400">
                                    <IconMicroscope className="w-12 h-12 mx-auto mb-3 text-amber-500" />
                                    <p>No scan history yet.</p>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    {results.map((result, idx) => (
                                        <div key={idx} className="p-4 border border-amber-700 rounded-xl bg-gray-900 shadow-sm hover:shadow-md transition">

                                            {/* TOP ROW */}
                                            <div className="flex justify-between items-start mb-2">
                                                <div className="flex items-center space-x-4">
                                                    <IconLeaf
                                                        className={`h-8 w-8 ${result.prediction === 'Healthy' ? 'text-amber-500' : 'text-red-500'}`}
                                                    />
                                                    <div>
                                                        <p className="text-lg font-semibold">{result.prediction}</p>
                                                        <p className="text-xs text-amber-400">Scanned: {result.timestamp}</p>
                                                    </div>
                                                </div>

                                                <div className="flex flex-col items-end space-y-2">
                                                    <div>
                                                        <p className="text-2xl font-bold text-amber-400">
                                                            {(result.confidence * 100).toFixed(1)}%
                                                        </p>
                                                        <p className="text-xs text-amber-400">Confidence</p>
                                                    </div>

                                                    {result.scan_id && (
                                                        <button
                                                            aria-label="Delete scan"
                                                            onClick={() => handleDeleteScan(result.scan_id)}
                                                            className="bg-red-700 hover:bg-red-600 text-white p-2 rounded-full shadow-md transition-transform duration-200 hover:scale-110"
                                                        >
                                                            <Trash2 className="w-4 h-4" />
                                                        </button>
                                                    )}
                                                </div>
                                            </div>

                                            {/* MESSAGE */}
                                            {result.message && (
                                                <div className="mt-2 p-3 bg-gray-800 border border-amber-700 rounded text-sm text-amber-300">
                                                    {result.message}
                                                </div>
                                            )}

                                            {/* RECOMMENDATION */}
                                            {result.recommendation && (
                                                <div className="mt-3 p-3 bg-gray-900 border border-amber-700 rounded text-sm text-amber-300">
                                                    <h4 className="font-semibold mb-1">Recommendation</h4>
                                                    <p>{result.recommendation}</p>
                                                </div>
                                            )}

                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                    </div>

                </div>
            </div>
        </>
    );
};

export default DashboardPage;
