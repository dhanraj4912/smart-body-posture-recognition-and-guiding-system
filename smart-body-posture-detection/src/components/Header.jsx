import React from "react";
import { FaRunning } from "react-icons/fa";
import { Link } from "react-router-dom";

function Header() {
  return (
    <nav className="bg-gray-950 text-white flex justify-between items-center px-12 h-20 fixed w-full top-0 z-50">
      <div className="text-2xl font-bold flex items-center gap-2">
        <FaRunning />
        <span>
          Posture<span className="text-yellow-400">Care</span>
        </span>
      </div>

      <ul className="flex items-center space-x-6">
        <li className="ml-4 text-2xl">
          <Link to="/">Home</Link>
        </li>
        <li className="ml-4 text-2xl">
          <a href="#features">Features</a>
        </li>
        <li className="ml-4 text-2xl">
          <a href="#how">How it Works</a>
        </li>
        <li className="ml-4 text-2xl">
          <Link to="/login">Sign in</Link>
        </li>
        <li className="ml-4 text-2xl">
          <Link to="/register">Register</Link>
        </li>
      </ul>
    </nav>
  );
}

export default Header;
