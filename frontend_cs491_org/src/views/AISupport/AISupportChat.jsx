import React, { useState } from 'react';
import { Box, TextField, Button, CircularProgress, Typography, Paper } from '@mui/material';
import axios from 'axios';
import { ROOTS_Flask } from '../../routes/paths'


function AISupportChat() {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleInputChange = (event) => {
        setInput(event.target.value);
    };

    const sendMessage = async () => {
        if (!input.trim()) return;  // Prevent sending empty messages
        const clinicianMessage = {
            author: 'clinician',
            content: input,
            type: 'text'
        };
        setMessages(messages => [...messages, clinicianMessage]); // Update the message list with the new message
        setInput(''); // Clear the input field
        setLoading(true); // Show loading indicator

        try {
            const response = await axios.post(`${ROOTS_Flask}/ai-help`, {
                clinical_question: clinicianMessage.content
            });
    
            // Construct separate messages for each part of the bot's response
            const picoMessage = {
                author: 'bot',
                content: (<Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                            PICO version of the question:
                        </Typography>),
                contentDetail: (<span>{response.data.pico_clinical_question}</span>),
                type: 'jsx'
            };
            
            const articleCountMessage = {
                author: 'bot',
                content: (<Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                            Number of relevant articles found in the PubMed database:
                        </Typography>),
                contentDetail: (<span>{response.data.article_count}</span>),
                type: 'jsx'
            };

            const articleTitlesMessage = {
                author: 'bot',
                content: (<Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                            Titles of the relevant articles:
                        </Typography>),
                contentDetail: (<ul>{response.data.article_titles.map((title, index) => (
                                <li key={index}>{title}</li>
                                ))}</ul>),
                type: 'jsx'
            };

            const aiSuggestionMessage = {
                author: 'bot',
                content: (<Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                            AI suggestion:
                        </Typography>),
                contentDetail: (<span>{response.data.RAG_GPT_output}</span>),
                type: 'jsx'
            };
    
            // Update the messages state with all the new messages
            setMessages(messages => [...messages, picoMessage, articleCountMessage, articleTitlesMessage, aiSuggestionMessage]);
        } catch (error) {
            setMessages(messages => [...messages, { author: 'bot', content: 'Sorry, AI could not help this time. Please try again...', type: 'text' }]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    };

    return (
        <Box sx={{
            p: 3,
            height: '90vh',
            display: 'flex',
            flexDirection: 'column',
            gap: 2
        }}>
            <Typography variant="h4" sx={{ textAlign: 'center' }}>AI Support</Typography>
            <Box sx={{
                flexGrow: 1,
                overflowY: 'auto'
            }}>
                {messages.map((message, index) => (
                    <Paper key={index} sx={{
                        margin: 1,
                        padding: 2,
                        textAlign: message.author === 'clinician' ? 'right' : 'left',
                        bgcolor: message.author === 'clinician' ? '#e0f7fa' : '#fce4ec',
                        overflowWrap: 'break-word',
                        boxSizing: 'border-box',
                        // Adjust the padding for the list inside bot messages
                        '& ul': {
                            padding: 0,
                            paddingLeft: theme => theme.spacing(2),
                            margin: 0
                        }
                    }}>
                        {message.type === 'jsx' ? (
                            <>
                                {message.content}
                                <Box sx={{ overflow: 'auto', paddingRight: 2 }}>
                                    {message.contentDetail}
                                </Box>
                            </>
                        ) : (
                            message.content
                        )}
                    </Paper>
                ))}
                {loading && <CircularProgress sx={{ display: 'block', margin: 'auto' }} />}
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TextField
                    fullWidth
                    variant="outlined"
                    label="Type your question"
                    value={input}
                    onChange={handleInputChange}
                    onKeyPress={handleKeyPress}
                    disabled={loading}
                />
                <Button variant="contained" onClick={sendMessage} disabled={!input.trim() || loading}>
                    Send
                </Button>
            </Box>
        </Box>
    );
}

export default AISupportChat;
