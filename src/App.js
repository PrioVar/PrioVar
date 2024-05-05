//import helix from './helix.svg';
//<img src={logo} className="App-logo" alt="logo" />
import './App.css';
import WelcomePage from "./components/WelcomePage";
import Team from "./components/Team";
import Contact from "./components/Contact";
import Appbar from './components/Appbar';
import PrioVar from './components/PrioVar';
import React   from "react";
import { motion, AnimatePresence } from 'framer-motion';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import './styles/fonts/index.css';
import Reports from './components/Reports';
import Resources from './components/Resources';

function App() {
  return (
    <Router>
      <div className="App">
      <motion.div
        className="App"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 1.0 }}
      >
        <Appbar />
        <AnimatePresence>
          <Routes>
            <Route path="/" element={<WelcomePage />} />
            <Route path="/team" element={<Team />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/resources" element={<Resources />} />
            <Route path="/contact-us" element={<Contact />} />
            <Route path="/priovar" element={<PrioVar/>} />
          </Routes>
        </AnimatePresence>
        </motion.div>
      </div>
    </Router>
  );
}

export default App;
