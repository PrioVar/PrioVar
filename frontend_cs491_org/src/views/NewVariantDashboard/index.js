import React from 'react';
import ReactDOM from 'react-dom';
import NewVariantDashboardTable from './NewVariantDashboardTable';
import { useParams } from 'react-router-dom';


function NewVariantDashboard() {
    const { fileName } = useParams();
    return (
        <div>
            <NewVariantDashboardTable fileName={fileName} />
        </div>
    );
}

export default NewVariantDashboard;
