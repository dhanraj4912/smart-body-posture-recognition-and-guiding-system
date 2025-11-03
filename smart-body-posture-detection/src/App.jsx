import React from 'react'
import Body from './components/Body'
import Header from './components/Header'
import Footer from './components/Footer'
import { Route, Routes } from 'react-router-dom'
import Login from './components/Login'
import Register from './components/Register'
import Dashboard from './components/Dashboard'


function App() {
  return (
    <div>
      <Header/>
      <div className="pt-20">
        <Routes>
          {/* Only render Body on home route */}
          <Route path="/" element={<Body />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register/>} />
          <Route path='/dashboard' element={<Dashboard/>}/>
        </Routes>
      </div>
    </div>


  )
}

export default App