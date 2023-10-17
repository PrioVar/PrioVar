import React, { useEffect } from 'react';
const WelcomePage = () => {
    useEffect(() => {
        document.title = 'PrioVar'
      }, []);
    return (
        <div
        style={{
            background: "#fefefe",
            padding: "20px",
            maxWidth: "960px",
            margin: "0 auto",
            textAlign: "left",
          }}
        >
        <h1>Project PrioVar</h1>
        <p>We introduce 'PrioVar,' where bioinformatics meet precision to transform rare disease diagnosis, and where health clinics become interconnected for a 
          collective impact. Our project is a cutting-edge genetic variant prioritization tool tailored for health clinics, designed to seamlessly integrate 
          phenotype data into the decision-making process. Clinics can prioritize genetic variants in order to make more accurate predictions about the causes 
          behind rare diseases, exchange valuable variant frequency information via our interface, and gain access to anonymous data from individuals with similar 
          phenotypes/genotypes with known diseases, all of which collectively revolutionize the detection and understanding of genetic conditions. Our mission is 
          simple: empower early and accurate diagnosis, ultimately improving the lives of those affected by rare diseases.</p>
        </div>
    )
};

export default WelcomePage;