import React, { useState, useCallback, useEffect } from 'react';
import { Upload, Loader2,  User, Settings, LogOut, ChevronDown, Trash2, Star, Zap } from 'lucide-react';
import type { DashboardPageProps, AnalysisResult } from '../../types';
import { API_BASE_URL } from '../../types';
import AlertMessage from '../ui/AlertMessage';
import { IconMicroscope, IconLeaf } from '../ui/Icons'; 

/* ========================================================================
   PLACEHOLDER DATA
======================================================================== */
const FARMER_TESTIMONIALS = [
    { name: "Aisha M.", rating: 5, quote: "The analysis is fast and the treatment recommendations saved my harvest! Truly a game-changer for my Coffee farm.", avatar: "AM" },
    { name: "John K.", rating: 4, quote: "Simple to use, even for an old farmer like me. The confidence score helps me trust the results.", avatar: "JK" },
    { name: "Sita P.", rating: 5, quote: "I can check my Coffee leaves right in the field. Essential tool for modern agriculture.", avatar: "SP" },
];

// --- PLACEHOLDER FOR YOUR ASSET ---
// **IMPORTANT**: REPLACE THIS WITH YOUR ACTUAL IMAGE IMPORT LATER
import  FARMER_HERO_IMAGE_URL from '../../assets/tech.jpg'; // Placeholder path

/* ========================================================================
   1. PROFILE DROPDOWN (Unchanged)
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
   2. DASHBOARD HEADER (Unchanged)
======================================================================== */
const DashboardHeader: React.FC<{ userEmail: string; userId: string | null; onLogout: () => void; userToken?: string | null }> = (props) => (
    <div className="fixed top-0 left-0 right-0 bg-gray-900/95 backdrop-blur-md py-4 px-5 shadow-xl z-50 border-b border-gray-800">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
            <div className="text-amber-400 text-xl sm:text-2xl font-extrabold flex items-center space-x-2">
                <IconLeaf className="text-amber-500 h-6 w-6" />
                <span>Coffee Scan | Dashboard</span>
            </div>
            <ProfileDropdown {...props} />
        </div>
    </div>
);

/* ========================================================================
   3. APP RATING CARD (Unchanged)
======================================================================== */
const AppRatingCard: React.FC = () => (
    <div className="bg-gray-800 p-6 rounded-2xl shadow-xl border-t-4 border-amber-500">
        <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-amber-300">Farmer Rating</h2>
            <p className="text-sm text-gray-400">based on 1.2K reviews</p>
        </div>
        <div className="flex items-center mt-4">
            <p className="text-6xl font-extrabold text-amber-400 mr-4">4.8</p>
            <div className="flex flex-col">
                <div className="flex space-x-1">
                    {[...Array(5)].map((_, i) => (
                        <Star key={i} className={`w-6 h-6 ${i < 4 ? 'text-amber-500 fill-amber-500' : (i === 4 ? 'text-amber-500 fill-amber-500/50' : 'text-gray-600')}`} />
                    ))}
                </div>
                <p className="text-sm mt-1 text-amber-300">Excellent Tool</p>
            </div>
        </div>
        <p className="mt-4 text-sm text-amber-400 border-t border-gray-700 pt-3">
            Join thousands of satisfied farmers protecting their crops.
        </p>
    </div>
);


/* ========================================================================
   4. FARMER TESTIMONIALS (Unchanged)
======================================================================== */
const FarmerVoice: React.FC = () => (
    <div className="bg-gray-800 p-6 rounded-2xl shadow-xl mt-8">
        <h2 className="text-xl sm:text-2xl font-bold mb-6 border-b border-amber-500 pb-2 flex items-center space-x-2">
            What The Farmers Say 🗣️
        </h2>
        <div className="space-y-6 max-h-96 overflow-y-auto pr-2">
            {FARMER_TESTIMONIALS.map((testimonial, index) => (
                <div key={index} className="bg-gray-900 p-4 rounded-xl border border-gray-700">
                    <div className="flex items-center mb-3">
                        <div className="w-8 h-8 rounded-full bg-amber-500 flex items-center justify-center text-gray-900 font-bold text-sm mr-3 flex-shrink-0">
                            {testimonial.avatar}
                        </div>
                        <div>
                            <p className="font-semibold text-amber-300">{testimonial.name}</p>
                            <div className="flex space-x-0.5 mt-0.5">
                                {[...Array(5)].map((_, i) => (
                                    <Star key={i} className={`w-3 h-3 ${i < testimonial.rating ? 'text-green-400 fill-green-400' : 'text-gray-600'}`} />
                                ))}
                            </div>
                        </div>
                    </div>
                    <blockquote className="text-sm italic text-amber-400 border-l-2 border-amber-600 pl-3">
                        "{testimonial.quote}"
                    </blockquote>
                </div>
            ))}
        </div>
    </div>
);

/* ========================================================================
   5. NEW VIEW: DASHBOARD HOME (Marketing/Welcome)
======================================================================== */
interface DashboardHomeProps {
    userEmail: string;
    onStartScan: () => void;
}

const DashboardHome: React.FC<DashboardHomeProps> = ({ userEmail, onStartScan }) => (
    <>
        {/* Welcome Hero Section */}
        <div className="grid md:grid-cols-2 bg-gray-900/90 rounded-2xl shadow-2xl border-l-4 border-amber-500 overflow-hidden mb-8">
            <div className="p-8">
                <h1 className="text-3xl sm:text-4xl font-extrabold text-amber-200 mb-3">
                    Hello, <span className="text-amber-500">{userEmail.split('@')[0]}</span>!
                </h1>
                <p className="text-lg text-amber-300 mb-6">
                    Ready to protect your harvest? Use Coffee Scan AI for instant, accurate disease diagnosis and treatment recommendations for your Coffee leaves.
                </p>
                
                <button
                    onClick={onStartScan}
                    className="w-full md:w-auto py-3 px-8 rounded-xl font-bold text-lg flex items-center justify-center space-x-3 transition-all 
                               bg-amber-500 hover:bg-amber-400 text-gray-900 shadow-lg hover:shadow-xl transform hover:scale-[1.02]"
                    aria-label="Start New Scan Page"
                >
                    <Zap className="h-6 w-6" />
                    <span>Start New Scan</span>
                </button>
            </div>
            {/* Placeholder Image */}
            <div className="hidden md:block">
                <img 
                    src={FARMER_HERO_IMAGE_URL} 
                    alt="Farmer inspecting healthy crops" 
                    className="w-full h-full object-cover"
                />
            </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
            {/* Left Column: Rating Card */}
            <AppRatingCard />

            {/* Right Column: Testimonials */}
            <FarmerVoice />
        </div>
    </>
);

/* ========================================================================
   6. NEW VIEW: SCAN PAGE (Upload and History)
======================================================================== */

interface ScanPageProps {
    userEmail: string;
    userToken: string | null;
    results: AnalysisResult[];
    setResults: React.Dispatch<React.SetStateAction<AnalysisResult[]>>;
    onBack: () => void;
    handleDeleteScan: (scanId: string | number | undefined) => Promise<void>;
    fetchSavedScans: () => Promise<void>;
}

const ScanPage: React.FC<ScanPageProps> = ({ 
    userEmail, 
    userToken, 
    results, 
    setResults, 
    onBack,
    handleDeleteScan,
    fetchSavedScans 
}) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [uploadMessage, setUploadMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    /* --- FILE UPLOAD --- */
    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files?.length) {
            const file = event.target.files[0];

            if (file.size > 5 * 1024 * 1024) {
                setUploadMessage({ text: 'File too large. Max 5MB.', type: 'error' });
                return;
            }

            if (previewUrl) URL.revokeObjectURL(previewUrl); 

            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
            setUploadMessage(null);
        }
    };

    /* --- UPLOAD AND PREDICT --- */
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
                headers: { 
                    ...(userToken ? { Authorization: `Bearer ${userToken}` } : {}) 
                },
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
                image: previewUrl || undefined, 
                scan_id: data.save_status === 'SAVED_SUCCESS' ? data.scan_id : undefined
            };

            if (data.save_status === 'SAVED_SUCCESS') {
                await fetchSavedScans();
                setUploadMessage({ text: 'Scan saved to history.', type: 'success' });
            } else {
                setResults(prev => [newResult, ...prev]);
            }

            setSelectedFile(null);
            if (previewUrl) URL.revokeObjectURL(previewUrl);
            setPreviewUrl(null);

        } catch {
            setUploadMessage({ text: 'Network error. Could not reach the server.', type: 'error' });

        } finally {
            setIsLoading(false);
        }
    }, [selectedFile, userToken, userEmail, fetchSavedScans, previewUrl, setResults]);

    return (
        <div className="grid lg:grid-cols-3 gap-8">
            
            {/* Left Column: Upload */}
            <div className="lg:col-span-1 space-y-8">
                <div className="bg-gray-800/90 p-6 rounded-2xl shadow-xl h-fit sticky top-[90px] border border-gray-700">
                    <div className="flex justify-between items-center mb-4 border-b border-amber-500 pb-2">
                        <h2 className="text-xl sm:text-2xl font-bold">New Scan</h2>
                        <button 
                            onClick={onBack}
                            className="text-sm text-amber-500 hover:text-amber-400 px-3 py-1 border border-amber-500/50 rounded-lg transition-colors"
                        >
                            ← Back to Home
                        </button>
                    </div>
                        
                    <AlertMessage message={uploadMessage?.text || null} type={uploadMessage?.type || null} />

                    <form onSubmit={handleUpload}>
                        <div className="mb-4">
                            <label className="block text-sm font-medium mb-2">Select Coffee Leaf Image (JPG/PNG)</label>

                            <label
                                htmlFor="file-upload"
                                className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-amber-500 rounded-xl cursor-pointer bg-gray-900/80 hover:bg-gray-800/90 transition-colors"
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
                                    aria-label="Select Coffee leaf image"
                                />
                            </label>
                        </div>

                        <button
                            type="submit"
                            disabled={isLoading || !selectedFile}
                            className={`w-full py-3 rounded-xl font-semibold flex items-center justify-center space-x-2 transition-all ${
                                isLoading || !selectedFile
                                    ? 'bg-amber-700/50 cursor-not-allowed'
                                    : 'bg-amber-500 hover:bg-amber-400 text-gray-900'
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
                                    <Zap className="h-5 w-5" />
                                    <span>Scan</span>
                                </>
                            )}
                        </button>
                    </form>
                </div>
            </div>

            {/* Right Column: Scan History */}
            <div className="lg:col-span-2 bg-gray-800/90 p-6 rounded-2xl shadow-xl border border-gray-700">
                <h2 className="text-xl sm:text-2xl font-bold mb-6 border-b border-amber-500 pb-2">
                    Scan History ({results.length})
                </h2>

                {results.length === 0 ? (
                    <div className="text-center py-12 bg-gray-900/80 rounded-xl text-amber-400 border border-gray-700">
                        <IconMicroscope className="w-12 h-12 mx-auto mb-3 text-amber-500" />
                        <p>No scan history yet. Upload an image to start.</p>
                    </div>
                ) : (
                    <div className="space-y-4 max-h-[80vh] overflow-y-auto pr-2">
                        {results.map((result, idx) => (
                            <div key={idx} className="p-4 border border-amber-700 rounded-xl bg-gray-900/80 shadow-lg hover:shadow-xl transition">

                                {/* TOP ROW */}
                                <div className="flex justify-between items-start mb-2">
                                    <div className="flex items-center space-x-4">
                                        {result.image && (
                                            <img src={result.image} alt="Scanned Leaf" className="w-12 h-12 object-cover rounded-md flex-shrink-0 border border-amber-600" />
                                        )}
                                        <div>
                                            <p className="text-lg font-bold text-amber-500">{result.prediction}</p>
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
                                    <div className="mt-2 p-3 bg-gray-700/80 border border-amber-700 rounded text-sm text-amber-300">
                                        {result.message}
                                    </div>
                                )}

                                {/* RECOMMENDATION */}
                                {result.recommendation && (
                                    <div className="mt-3 p-3 bg-gray-900/80 border border-amber-700 rounded text-sm text-amber-300">
                                        <h4 className="font-semibold mb-1">Recommendation</h4>
                                        <p>{result.recommendation}</p>
                                    </div>
                                )}
                                {/* Image Preview */}
                                {result.image && (
                                    <div className="mt-3">
                                        <a href={result.image} target="_blank" rel="noopener noreferrer" className="text-xs text-amber-500 hover:text-amber-400 underline">
                                            View Scanned Image
                                        </a>
                                    </div>
                                )}

                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};


/* ========================================================================
   7. MAIN DASHBOARD PAGE (Router)
======================================================================== */
const DashboardPage: React.FC<DashboardPageProps> = ({ userToken, userId, userEmail, onLogout }) => {
    useEffect(() => {
        // Existing style cleanup
        const originalMargin = document.body.style.margin;
        const originalPadding = document.body.style.padding;
        document.body.style.margin = '0';
        document.body.style.padding = '0';
        return () => {
            document.body.style.margin = originalMargin;
            document.body.style.padding = originalPadding;
        };
    }, []);

    const [results, setResults] = useState<AnalysisResult[]>([]);
    const [currentView, setCurrentView] = useState<'home' | 'scan'>('home'); // State to manage views
    const [, setIsFetchingHistory] = useState(false);

    
    /* --- DELETE SCAN (Centralized) --- */
    const handleDeleteScan = useCallback(async (scanId: string | number | undefined) => {
        if (!scanId) return;

        // Simplified deletion logic for brevity
        try {
            const res = await fetch(`${API_BASE_URL}/delete_scan/${scanId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    ...(userToken ? { Authorization: `Bearer ${userToken}` } : {})
                }
            });

            if (!res.ok) throw new Error("Delete failed");

            setResults(prev => prev.filter(r => r.scan_id !== scanId));
            // You can add a transient success message here if needed
        } catch (e) {
            console.error("Error deleting scan:", e);
        }
    }, [userToken]);

    /* --- FETCH SAVED SCANS (Centralized) --- */
    const fetchSavedScans = useCallback(async () => {
        if (!userEmail) return setResults([]);

        setIsFetchingHistory(true);
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
            setIsFetchingHistory(false);
        }
    }, [userEmail, userToken]);

    useEffect(() => {
        // Fetch history immediately upon entering the dashboard, regardless of the starting view
        if (userEmail) fetchSavedScans();
    }, [userEmail, fetchSavedScans]);

    /* --- PAGE UI RENDER --- */
    return (
        <>
            <DashboardHeader userEmail={userEmail} userId={userId} userToken={userToken} onLogout={onLogout} />
            
            {/* Background Container */}
            <div className="min-h-screen pt-[4.5rem] p-4 sm:p-8 text-amber-300 relative 
                bg-fixed bg-cover bg-center 
                bg-[url('https://images.unsplash.com/photo-1549419137-b6f12089f213?ixlib=rb-4.0.3&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=1920&h=1080&fit=crop')]"> 
                
                {/* Overlay for readability */}
                <div className="absolute inset-0 bg-gray-900/80 backdrop-blur-sm"></div>

                <div className="max-w-7xl mx-auto relative z-10">
                    {/* View Switch */}
                    {currentView === 'home' ? (
                        <DashboardHome 
                            userEmail={userEmail} 
                            onStartScan={() => setCurrentView('scan')} 
                        />
                    ) : (
                        <ScanPage
                            userEmail={userEmail}
                            userToken={userToken}
                            results={results}
                            setResults={setResults}
                            onBack={() => setCurrentView('home')}
                            handleDeleteScan={handleDeleteScan}
                            fetchSavedScans={fetchSavedScans}
                        />
                    )}
                </div>
            </div>
        </>
    );
};

export default DashboardPage;