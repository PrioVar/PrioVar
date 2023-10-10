import React, { useState, useEffect } from 'react';
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
        <form action="/" method="GET">
          <input
            type="submit"
            className="navbar-brand"
            style={{
              background: 'none',
              border: 'none',
              marginLeft: '2rem',
              padding: 0,
              cursor: 'pointer',
              fontSize: '1.5rem',
              fontWeight: 'bold',
              transition: 'font-size 0.2s ease',
            }}
            value="PrioVar"
          />
        </form>
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
            <NavItem to="/" label="Project"  style={{ fontWeight: 'bold' }}/>
            <NavItem to="/team" label="Team"  style={{ fontWeight: 'bold' }}/>
            <NavItem to="/reports" label="Reports"  style={{ fontWeight: 'bold' }}/>
            <NavItem to="/resources" label="Resources Used"  style={{ fontWeight: 'bold' }}/>
            <NavItem to="/contact-us" label="Contact Us"  style={{ fontWeight: 'bold' }}/>
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
          <NavItem to="/" label="Project"  style={{ fontWeight: 'bold' }}/>
            <NavItem to="/team" label="Team"  style={{ fontWeight: 'bold' }}/>
            <NavItem to="/reports" label="Reports"  style={{ fontWeight: 'bold' }}/>
            <NavItem to="/resources" label="Resources Used"  style={{ fontWeight: 'bold' }}/>
            <NavItem to="/contact-us" label="Contact Us"  style={{ fontWeight: 'bold' }}/>
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

const NavItem = ({ to, label , style}) => {
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
      <input
        type="submit"
        style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer', ...style }}
        value={label}
      />
    </form>
  );
};

export default Appbar;