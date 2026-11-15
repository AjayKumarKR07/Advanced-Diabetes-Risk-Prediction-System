/**
 * src/App.jsx
 * ===========
 * Root application with React Router routes, AuthProvider,
 * PublicLayout (Navbar + Footer) and AuthenticatedLayout (Persistent Sidebar + Header).
 * Phase 6.11: Professional Authenticated Application Layout & User Experience.
 * Phase 6.16.1: Route-based code splitting and lazy loading for high performance.
 */

import { lazy, Suspense } from 'react'
import { Routes, Route, Outlet } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import { ProtectedRoute, GuestOnlyRoute } from './components/ProtectedRoute'
import AuthenticatedLayout from './components/AuthenticatedLayout'
import Navbar from './components/Navbar'
import Footer from './components/Footer'

// Lazy loaded page components for optimal initial bundle size and split routing
const Landing = lazy(() => import('./pages/Landing'))
const Dashboard = lazy(() => import('./pages/Dashboard'))
const HealthAssessmentForm = lazy(() => import('./pages/HealthAssessmentForm'))
const PredictionResult = lazy(() => import('./pages/PredictionResult'))
const ModelPerformance = lazy(() => import('./pages/ModelPerformance'))
const AboutMethodology = lazy(() => import('./pages/AboutMethodology'))
const History = lazy(() => import('./pages/History'))
const Profile = lazy(() => import('./pages/Profile'))
const Settings = lazy(() => import('./pages/Settings'))
const Register = lazy(() => import('./pages/Register'))
const Login = lazy(() => import('./pages/Login'))

/**
 * Lightweight loading spinner fallback while lazy chunks load.
 */
function PageLoadingFallback() {
  return (
    <div className="flex-1 flex items-center justify-center min-h-[50vh]">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-3 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
        <span className="text-xs font-medium text-slate-400">Loading module...</span>
      </div>
    </div>
  )
}

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
      <Suspense fallback={<PageLoadingFallback />}>
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
      </Suspense>
    </AuthProvider>
  )
}
