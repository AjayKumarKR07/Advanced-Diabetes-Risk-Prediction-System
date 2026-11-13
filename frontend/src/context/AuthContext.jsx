/**
 * src/context/AuthContext.jsx
 * ===========================
 * React Context providing global user authentication state and actions.
 * Automatically restores session on page reload and validates token with /api/auth/me.
 */

import { createContext, useContext, useEffect, useState } from 'react'
import {
  clearAuth,
  fetchCurrentUser,
  getStoredToken,
  getStoredUser,
  loginUser,
  registerUser,
  storeAuth,
} from '../services/auth'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(getStoredUser())
  const [token, setToken] = useState(getStoredToken())
  const [loading, setLoading] = useState(true)

  // Verify session on mount if token exists
  useEffect(() => {
    const savedToken = getStoredToken()
    if (!savedToken) {
      setLoading(false)
      return
    }

    fetchCurrentUser()
      .then((userData) => {
        setUser(userData)
        setToken(savedToken)
      })
      .catch(() => {
        // Token invalid or expired — clear stored session
        clearAuth()
        setUser(null)
        setToken(null)
      })
      .finally(() => {
        setLoading(false)
      })
  }, [])

  const login = async (email, password, rememberMe = true) => {
    const data = await loginUser({ email, password })
    storeAuth(data.access_token, data.user, rememberMe)
    setToken(data.access_token)
    setUser(data.user)
    return data
  }

  const register = async (fullName, email, password) => {
    const data = await registerUser({
      full_name: fullName,
      email,
      password,
    })
    storeAuth(data.access_token, data.user, true)
    setToken(data.access_token)
    setUser(data.user)
    return data
  }

  const logout = () => {
    clearAuth()
    setUser(null)
    setToken(null)
  }

  const updateUser = (updatedUser) => {
    setUser(updatedUser)
  }

  const value = {
    user,
    token,
    loading,
    isAuthenticated: Boolean(user && token),
    login,
    register,
    logout,
    updateUser,
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

