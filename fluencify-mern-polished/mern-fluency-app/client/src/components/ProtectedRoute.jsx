import { Navigate } from "react-router-dom";
import { currentUser } from "../api/auth.js";

export default function ProtectedRoute({ children }) {
  const user = currentUser();
  if (!user || !localStorage.getItem("token")) {
    return <Navigate to="/login" replace />;
  }
  return children;
}
