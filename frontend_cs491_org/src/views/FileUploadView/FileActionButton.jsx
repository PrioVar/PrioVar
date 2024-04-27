import React from 'react';
import Button from '@mui/material/Button';
import Tooltip from '@mui/material/Tooltip';

const FileActionButton = ({ fileStatus }) => {
    const isAnnotated = fileStatus === 'FILE_ANNOTATED';

    return (
        <Tooltip title={isAnnotated ? "File already annotated" : "Click to annotate file"}>
            <span> {/* Tooltip children need to be able to hold a ref */}
                <Button 
                    disabled={isAnnotated}
                    onClick={() => {
                        if (!isAnnotated) {
                            console.log('Annotate the file!');
                        }
                    }}
                    variant="contained"
                    color="primary"
                >
                    Annotate File
                </Button>
            </span>
        </Tooltip>
    );
};

export default FileActionButton;
