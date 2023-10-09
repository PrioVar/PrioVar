import React, { useEffect } from 'react';
const WelcomePage = () => {
    useEffect(() => {
        document.title = 'Xaga'
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
        <h1>Project Xaga</h1>
        <p>Introducing 'PrioVar,' where science meets precision to transform rare disease diagnosis. 
            We leverage cutting-edge analysis of phenotype and genotype data to prioritize variants, 
            providing medical professionals with actionable insights. Our mission is simple: empower early 
            and accurate diagnosis, ultimately improving the lives of those affected by rare diseases.</p>
        <br/>
        <h2>Details of Xaga</h2>
        <p>Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, 
            eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam 
            voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione 
            voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit,
            sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad 
            minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi 
            consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, 
            vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?</p>
        <p>
        Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Integer suscipit id purus nec 
        semper. Mauris scelerisque, turpis vel tempor accumsan, justo diam venenatis lacus, in vehicula turpis ex congue magna. 
        Suspendisse id finibus ipsum. Morbi fermentum velit ex, sed sagittis lorem consectetur vitae. Curabitur mi enim, fringilla 
        a nibh et, volutpat tincidunt erat. Pellentesque sit amet purus ut metus luctus laoreet. Suspendisse nunc eros, facilisis 
        sed condimentum ac, suscipit vitae nibh. Nullam rutrum, ex et consectetur imperdiet, justo nibh vehicula ex, a euismod dui 
        tellus quis ante. Cras euismod, odio quis imperdiet finibus, libero augue porta nunc, in tempor enim lacus viverra magna. 
        Nunc eu velit a nulla porttitor mollis vel sed magna. Phasellus aliquet efficitur turpis vitae tincidunt. Aliquam vitae 
        ultrices metus. Sed tincidunt risus et risus ultrices aliquet. Fusce feugiat vel lacus rutrum euismod. 
        </p>
        </div>
    )
};

export default WelcomePage;