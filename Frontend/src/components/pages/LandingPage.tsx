import React, { useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import type { LandingPageProps } from '../../types';
import AlertMessage from '../ui/AlertMessage';
import { IconUploadInternal, IconMicroscope, IconCheckCircleInternal, Grid } from '../ui/Icons';
import DelayedLink from '../ui/DelayedLink';

// Placeholder for your actual background image URL.
import HERO_BG_IMAGE_URL from '../../assets/coffee.jpg'; 

// --- Navigation Bar Component ---
// This component is fixed at the top, potentially covering content.
const NavigationBar: React.FC = () => (
    <nav className="fixed top-0 left-0 right-0 z-50 py-4 px-8 bg-black/50 backdrop-blur-md shadow-lg">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
            <h1 className="text-2xl font-bold text-amber-500 font-roboto ">
              <link rel="stylesheet" href="coffeescan-ai.netlify.app/" />
               <Link to="/" className="hover:text-yellow-400 transition">  Kahawa Bora </Link>
            </h1>
            <div className="flex items-center space-x-6 text-white font-semibold">
                {/* Internal Scroll Links */}
                <Link to="/#how-it-works" className="hover:text-yellow-400 transition">How It Works</Link>
                <Link to="/#features" className="hover:text-yellow-400 transition">Features</Link>
                <Link to="/#features" className="hover:text-yellow-400 transition">Why Us</Link> 
                <Link to="/contact" className="hover:text-yellow-400 transition">Contact</Link> 
                
                {/* Auth Buttons */}
                <Link to="/login" className="px-4 py-2 border border-gray-400 rounded-lg hover:bg-gray-700 transition">Login</Link>
                <Link to="/register" className="bg-amber-500 text-black px-4 py-2 rounded-lg hover:bg-yellow-400 transition">Register</Link>
            </div>
        </div>
    </nav>
);
// --- End Navigation Bar Component ---


const LandingPage: React.FC<LandingPageProps> = ({ message, setMessage }) => {
    const location = useLocation();

    // The fixed navigation bar appears to be about 64px (h-16) to 80px tall including padding.
    // We will use 90px as a safe offset to clear the navbar completely.
    const NAVBAR_HEIGHT_OFFSET = 90; 

    useEffect(() => {
        if (message.message) {
            const timer = setTimeout(() => setMessage({ message: null, type: null }), 5000);
            return () => clearTimeout(timer);
        }
    }, [message, setMessage]);

    // FIX: Anchor Scrolling Logic
    useEffect(() => {
        if (location.hash) {
            const element = document.getElementById(location.hash.substring(1));
            if (element) {
                setTimeout(() => {
                    // FIX: Use the calculated NAVBAR_HEIGHT_OFFSET
                    const yOffset = -NAVBAR_HEIGHT_OFFSET; 
                    const y = element.getBoundingClientRect().top + window.pageYOffset + yOffset;
                    
                    window.scrollTo({ top: y, behavior: 'smooth' });
                }, 0);
            }
        } else {
            // Scroll to top when loading the root path without hash
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }, [location]);

  return (
    <div className="min-h-full bg-black">
      
      <NavigationBar /> {/* Included for testing navigation links */}

      {message.message && (
        <div className="fixed top-20 right-4 z-50">
          <AlertMessage message={message.message} type={message.type} />
        </div>
      )}

      {/* Hero Section - Full Page Background */}
      <header 
        className="relative min-h-screen flex items-center justify-center text-white text-center"
        style={{ height: '100vh'}} // Height remains 100vh
      >
        {/* Background Image Container */}
        <div 
          className="absolute inset-0 bg-cover bg-center"
          style={{ 
            backgroundImage: `url('${HERO_BG_IMAGE_URL}')`,
            filter: 'contrast(105%) brightness(95%)',
          }}
        >
          <div className="absolute inset-0 bg-black/50"></div> 
        </div>

        {/* Hero Content (z-index 10 to ensure it's above the background) */}
        {/* FIX: Increased padding-top to ensure content starts below the fixed navbar */}
        <div className="relative z-10 pt-[120px] pb-12 px-8 max-w-4xl mx-auto"> 
          <p className="text-sm uppercase tracking-widest text-amber-300 mb-2 font-semibold">
            Future-Proof Your Harvest
          </p>
          <h1 className="text-5xl sm:text-7xl font-extrabold mb-4 leading-tight">
            Stop Coffee Disease <br /> <span className="text-yellow-400">Before It Spreads</span>
          </h1>
          <p className="text-xl text-amber-100 mb-10 max-w-3xl mx-auto font-light">
            Empower your farm with instant, AI-driven diagnostics for common coffee plant ailments, delivered right to your phone.
          </p>
          
          <div className="flex flex-col md:flex-row justify-center items-center space-y-4 md:space-y-0 md:space-x-6 mb-10">
            <DelayedLink 
              to="/login" 
              className="bg-amber-500 text-black font-bold py-4 px-10 rounded-xl text-lg shadow-2xl hover:bg-yellow-400 transition transform hover:scale-[1.02] flex items-center justify-center space-x-2 w-full md:w-auto"
              delayMs={500}
            >
              Start Instant Diagnosis
            </DelayedLink>
            <Link
              to="/#how-it-works"
              className="text-yellow-400 font-bold py-4 px-10 rounded-xl text-lg shadow-xl border-2 border-amber-400 bg-black/30 hover:bg-black/50 transition flex items-center justify-center w-full md:w-auto"
            >
              See How It Works →
            </Link>
          </div>

          <div className="flex flex-wrap justify-center gap-6 mt-12 pt-4 border-t border-amber-400/30">
            <span className="text-sm font-semibold text-white flex items-center">
              <IconCheckCircleInternal className="h-4 w-4 text-amber-400 mr-1" /> Adaptable
            </span>
            <span className="text-sm font-semibold text-white flex items-center">
              <IconCheckCircleInternal className="h-4 w-4 text-amber-400 mr-1" /> Convenient & Fast
            </span>
            <span className="text-sm font-semibold text-white flex items-center">
              <IconCheckCircleInternal className="h-4 w-4 text-amber-400 mr-1" /> Reliable Accuracy
            </span>
          </div>
        </div>
      </header>

      {/* How It Works Section - Target ID for anchor scrolling */}
      <section id="how-it-works" className="py-24 bg-brown-900 border-b border-brown-800">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-4xl font-extrabold text-center text-yellow-400 mb-16">
             The 3-Step Protection Plan
          </h2>
          
          <div className="flex flex-col lg:flex-row items-stretch justify-between gap-8 p-6 rounded-2xl bg-brown-800/80 shadow-2xl border-l-8 border-amber-400">
            {/* Steps 1, 2, 3 content remains the same */}
            <div className="flex-1 space-y-4">
                <span className="text-5xl font-black text-amber-400">1.</span>
                <h3 className="text-2xl font-bold text-yellow-400 flex items-center space-x-2">
                    <IconUploadInternal className="h-7 w-7 text-amber-400" /> <span>Capture & Upload</span>
                </h3>
                <p className="text-amber-200">
                    Simply take a clear photo of the affected coffee leaf with your phone. Our platform handles the rest.
                </p>
            </div>
            <div className="flex-1 space-y-4">
                <span className="text-5xl font-black text-amber-400">2.</span>
                <h3 className="text-2xl font-bold text-yellow-400 flex items-center space-x-2">
                    <IconMicroscope className="h-7 w-7 text-amber-400" /> <span>AI Analysis</span>
                </h3>
                <p className="text-amber-200">
                    Our specialized AI model instantly detects common diseases (Rust, Phoma, Miner, etc.) and provides a high-confidence diagnosis.
                </p>
            </div>
            <div className="flex-1 space-y-4">
                <span className="text-5xl font-black text-amber-400">3.</span>
                <h3 className="text-2xl font-bold text-yellow-400 flex items-center space-x-2">
                    <IconCheckCircleInternal className="h-7 w-7 text-amber-400" /> <span>Actionable Advice</span>
                </h3>
                <p className="text-amber-200">
                    Receive simple, actionable steps for immediate treatment and prevention, helping you save your yields.
                </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section - Target ID for anchor scrolling (Why Us and Features links) */}
      <section id="features" className="py-24 bg-black border-b border-brown-800">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <h2 className="text-4xl font-extrabold text-yellow-400 mb-4">Farm Smarter, Not Harder</h2>
          <p className="text-lg text-amber-300 mb-16 max-w-3xl mx-auto">
            Key benefits that make CoffeeScanAI essential for every coffee farmer.
          </p>

          <div className="grid md:grid-cols-3 gap-8 text-left">
            {/* Feature content remains the same */}
            <div className="p-6 bg-brown-800 rounded-xl shadow-xl border-t-4 border-amber-400 hover:shadow-2xl transition text-white">
                <Grid className="w-8 h-8 text-amber-400 mb-3" />
                <h3 className="text-xl font-bold mb-2">Identify Early Threats</h3>
                <p className="text-amber-200">Catch diseases at their earliest stages, minimizing damage and costly wide-scale treatment.</p>
            </div>
            <div className="p-6 bg-brown-800 rounded-xl shadow-xl border-t-4 border-yellow-400 hover:shadow-2xl transition text-white">
                <IconMicroscope className="w-8 h-8 text-yellow-400 mb-3" />
                <h3 className="text-xl font-bold mb-2">High Accuracy, Low Error</h3>
                <p className="text-amber-200">Leverage our model trained on local coffee leaf images for high diagnostic reliability.</p>
            </div>
            <div className="p-6 bg-brown-800 rounded-xl shadow-xl border-t-4 border-amber-500 hover:shadow-2xl transition text-white">
                <IconCheckCircleInternal className="w-8 h-8 text-amber-500 mb-3" />
                <h3 className="text-xl font-bold mb-2">Maximize Yields</h3>
                <p className="text-amber-200">Protect your investment by getting the right diagnosis and effective treatment plan immediately.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action Section */}
      <section id="cta" className="py-24 bg-brown-900">
        <div className="max-w-7xl mx-auto px-4 text-center">
          {/* CTA content remains the same */}
          <h2 className="text-4xl font-extrabold text-yellow-400 mb-4">Ready to Protect Your Investment?</h2>
          <p className="text-lg text-amber-300 mb-8">
            Join the coffee farmers using technology to secure their harvest against disease.
          </p>
          <DelayedLink 
            to="/register" 
            className="inline-block py-4 px-12 bg-amber-500 text-black font-semibold rounded-xl text-lg hover:bg-yellow-400 transition transform hover:scale-[1.05]"
            delayMs={500}
          >
            Create Your Free Account
          </DelayedLink>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;