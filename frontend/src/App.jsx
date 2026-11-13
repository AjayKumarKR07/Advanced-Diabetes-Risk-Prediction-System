/**
 * src/App.jsx
 * ===========
 * Root application with React Router routes, AuthProvider,
 * PublicLayout (Navbar + Footer) and AuthenticatedLayout (Persistent Sidebar + Header).
 * Phase 6.11: Professional Authenticated Application Layout & User Experience.
 */

import { Routes, Route, Outlet } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import { ProtectedRoute, GuestOnlyRoute } from './components/ProtectedRoute'
import AuthenticatedLayout from './components/AuthenticatedLayout'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import HealthAssessmentForm from './pages/HealthAssessmentForm'
import PredictionResult from './pages/PredictionResult'
import ModelPerformance from './pages/ModelPerformance'
import AboutMethodology from './pages/AboutMethodology'
import History from './pages/History'
import Profile from './pages/Profile'
import Settings from './pages/Settings'
import Register from './pages/Register'
import Login from './pages/Login'

/**
 * Public Layout with sticky top Navbar and persistent Footer.
 * Used for public landing, research performance catalog, methodology, and guest auth pages.
 */
function PublicLayout() {
  return (
    <div className="min-h-screen bg-slate-50 flex flex-col justify-between selection:bg-blue-100 selection:text-blue-900">
      <Navbar />
      {/* Top padding to clear fixed navbar */}
      <main className="pt-16 flex-1 flex flex-col">
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        {/* Public & Guest Routes */}
        <Route element={<PublicLayout />}>
          <Route path="/" element={<Landing />} />
          <Route path="/performance" element={<ModelPerformance />} />
          <Route path="/about" element={<AboutMethodology />} />

          {/* Guest-Only Authentication Routes */}
          <Route
            path="/register"
            element={
              <GuestOnlyRoute>
                <Register />
              </GuestOnlyRoute>
            }
          />
          <Route
            path="/login"
            element={
              <GuestOnlyRoute>
                <Login />
              </GuestOnlyRoute>
            }
          />

          {/* 404 fallback */}
          <Route
            path="*"
            element={
              <div className="max-w-xl mx-auto text-center py-32 px-4 flex-1 flex flex-col items-center justify-center">
                <div className="text-7xl font-black text-slate-200 mb-3 tracking-tight">404</div>
                <h1 className="text-2xl font-bold text-slate-800 mb-2">Page Not Found</h1>
                <p className="text-slate-500 mb-8 max-w-sm">
                  The page you are looking for does not exist or has been moved.
                </p>
                <a href="/" className="btn-primary inline-flex items-center gap-2">
                  Return to Homepage
                </a>
              </div>
            }
          />
        </Route>

        {/* Authenticated Application Routes (Persistent Sidebar + Header Layout) */}
        <Route
          element={
            <ProtectedRoute>
              <AuthenticatedLayout />
            </ProtectedRoute>
          }
        >
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/assess" element={<HealthAssessmentForm />} />
          <Route path="/result" element={<PredictionResult />} />
          <Route path="/history" element={<History />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Routes>
    </AuthProvider>
  )
}
