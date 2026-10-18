import { Routes, Route, Link, Navigate, useNavigate } from "react-router-dom";
import Login from "./pages/Login.jsx";
import Signup from "./pages/Signup.jsx";
import Record from "./pages/Record.jsx";
import History from "./pages/History.jsx";
import ProtectedRoute from "./components/ProtectedRoute.jsx";
import { currentUser, logout } from "./api/auth.js";

export default function App() {
  const user = currentUser();
  const navigate = useNavigate();

  function handleLogout() {
    logout();
    navigate("/login");
  }

  return (
    <>
      <nav>
        <Link to="/">Fluency App</Link>
        <div className="spacer" />
        {user ? (
          <>
            <Link to="/record">Analyze</Link>
            <Link to="/history">History</Link>
            <span className="muted">{user.name}</span>
            <button onClick={handleLogout}>Log out</button>
          </>
        ) : (
          <>
            <Link to="/login">Log in</Link>
            <Link to="/signup">Sign up</Link>
          </>
        )}
      </nav>

      <Routes>
        <Route path="/" element={<Navigate to={user ? "/record" : "/login"} replace />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route
          path="/record"
          element={
            <ProtectedRoute>
              <Record />
            </ProtectedRoute>
          }
        />
        <Route
          path="/history"
          element={
            <ProtectedRoute>
              <History />
            </ProtectedRoute>
          }
        />
      </Routes>
    </>
  );
}
