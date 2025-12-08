export const API_BASE_URL = 'https://coffee-disease-detector.onrender.com' ; 
export const LOGIN_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

export type Page = 'landing' | 'auth' | 'dashboard';

export interface AlertMessageProps {
  message: string | null;
  type: 'success' | 'error' | null;
}

export interface AnalysisResult {
  filename: string;
  prediction: string;
  confidence: number;
  timestamp: string;
  status?: string;
  message?: string;
  recommendation?: string;
  scan_id?: string | number;
}

export interface AuthPageProps {
  onLoginSuccess: (userId: string, email: string, token?: string) => void;
}

export interface DashboardPageProps {
  userToken: string | null;
  userId: string | null;
  userEmail: string;
  onLogout: () => void;
}

export interface NavbarProps {
  setCurrentPage: (page: Page) => void;
  userToken: string | null;
  onLogout: () => void;
}

export interface LandingPageProps {
  setCurrentPage: (page: Page) => void;
  diseaseCategories: string[];
  message: AlertMessageProps;
  setMessage: (msg: AlertMessageProps) => void;
}