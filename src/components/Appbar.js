import React, { useState, useEffect } from 'react';
import { FaProjectDiagram, FaUsers, FaFileAlt, FaTools, FaPhone, FaDna } from 'react-icons/fa';
//import LanguageSelector from './LanguageSelector';

const Appbar = () => {
  const [isNavDisplayedMobile, setNavDisplayedMobile] = useState(false);
  const [isNavOpen, setNavOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  const handleToggle = () => {
    setNavOpen(!isNavOpen);
    setNavDisplayedMobile( isMobile && isNavOpen);
  };

  useEffect(() => {
    const checkScreenWidth = () => {
      setIsMobile(window.innerWidth < 768); // Adjust the breakpoint as needed
      setNavDisplayedMobile( isMobile && isNavOpen);
    };

    // Add event listener to check screen width on resize
    window.addEventListener('resize', checkScreenWidth);
    checkScreenWidth(); // Check screen width on initial load

    // Cleanup the event listener on unmount
    return () => {
      window.removeEventListener('resize', checkScreenWidth);
    };
  }, [isMobile, isNavOpen]);

  return (
    <nav style={{ backgroundColor: '#f8f9fa', padding: '12px 0' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', marginLeft: '2rem' }}>
          <img src="/PrioVar_logo.png" alt="PrioVar Logo" style={{ height: '40px', marginRight: '10px' }} />
          <form action="/" method="GET">
            <input
              type="submit"
              className="navbar-brand"
              style={{
                background: 'none',
                border: 'none',
                padding: 0,
                cursor: 'pointer',
                fontSize: '1.5rem',
                fontWeight: 'bold',
                transition: 'font-size 0.2s ease',
                fontFamily: 'CircularStd',
              }}
              value="PrioVar"
            />
          </form>
        </div>
        {isMobile ? (
          <button
            style={{ background: 'none', marginRight: '2rem', border: 'none', padding: 0, cursor: 'pointer', fontSize: '1.5rem' }}
            onClick={handleToggle}
          >
            <span style={{ display: 'none' }}>Toggle navigation</span>
            <span className="navbar-toggler-icon">☰</span>
          </button>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'flex-start', maxWidth: '960px', margin: '0 auto' }}>
            <NavItem to="/" label="Project" iconColor="#007BFF" Icon={FaProjectDiagram} style={{ fontWeight: 'bold', fontFamily: 'CircularStd', fontSize: '1rem' }}/>
            <NavItem to="/team" label="Team" iconColor="#0056B3" Icon={FaUsers} style={{ fontWeight: 'bold', fontFamily: 'CircularStd', fontSize: '1rem'  }}/>
            <NavItem to="/reports" label="Reports" iconColor="#6C757D" Icon={FaFileAlt} style={{ fontWeight: 'bold', fontFamily: 'CircularStd', fontSize: '1rem'  }}/>
            <NavItem to="/resources" label="Resources Used" iconColor="#343A40" Icon={FaTools} style={{ fontWeight: 'bold', fontFamily: 'CircularStd', fontSize: '1rem'  }}/>
            <NavItem to="/contact-us" label="Contact Us" iconColor="#004085" Icon={FaPhone} style={{ fontWeight: 'bold', fontFamily: 'CircularStd', fontSize: '1rem'  }}/>
            <NavItem to="/priovar" label="PrioVar" iconColor="#003E70" Icon={FaDna} style={{ fontWeight: 'bold', fontFamily: 'CircularStd', fontSize: '1rem'  }}/>
            {
              /* 
              <div style={{ display: 'flex', alignItems: 'center', marginLeft: '0.5rem', fontWeight: 'bold' }}>
              <LanguageSelector />
            </div>
             */
            }
          </div>
        )}
      </div>
      {isNavDisplayedMobile && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', maxWidth: '960px', margin: '0 auto' }}>
            <NavItem to="/" label="Project" iconColor="#007BFF" Icon={FaProjectDiagram} style={{ fontWeight: 'bold', fontFamily: 'CircularStd', fontSize: '1rem' }}/>
            <NavItem to="/team" label="Team" iconColor="#0056B3" Icon={FaUsers} style={{ fontWeight: 'bold', fontFamily: 'CircularStd', fontSize: '1rem'  }}/>
            <NavItem to="/reports" label="Reports" iconColor="#6C757D" Icon={FaFileAlt} style={{ fontWeight: 'bold', fontFamily: 'CircularStd', fontSize: '1rem'  }}/>
            <NavItem to="/resources" label="Resources Used" iconColor="#343A40" Icon={FaTools} style={{ fontWeight: 'bold', fontFamily: 'CircularStd', fontSize: '1rem'  }}/>
            <NavItem to="/contact-us" label="Contact Us" iconColor="#004085" Icon={FaPhone} style={{ fontWeight: 'bold', fontFamily: 'CircularStd', fontSize: '1rem'  }}/>
            <NavItem to="/priovar" label="PrioVar" iconColor="#003E70" Icon={FaDna} style={{ fontWeight: 'bold', fontFamily: 'CircularStd', fontSize: '1rem'  }}/>
            {
              /*
                <NavItem style = {{ className:'ml-auto' }}>
                  <div style={{ display: 'flex', alignItems: 'center', marginLeft: '0.5rem', fontWeight: 'bold' }}>
                  <LanguageSelector />
              </div>
              </NavItem>
              */
            }
        </div>
      )}

    </nav>
  );
};

const NavItem = ({ to, label, style, Icon, iconColor }) => {
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseEnter = () => {
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
  };

  return (
    <form
      action={to}
      method="GET"
      style={{
        padding: '0 8px',
        fontSize: isHovered ? '1.5rem' : '1rem',
        fontWeight: 'bold',
        transition: 'font-size 0.3s ease',
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {Icon && <Icon style={{ marginRight: '10px', fontSize: '1.5rem', color: iconColor }} />}
      <input
        type="submit"
        style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer', ...style }}
        value={label}
      />
    </form>
  );
};

export default Appbar;