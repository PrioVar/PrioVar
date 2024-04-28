import React, { useState } from 'react';
import { Box, TextField, Button, CircularProgress, Typography, Paper } from '@mui/material';
import axios from 'axios';
import { ROOTS_Flask } from '../../routes/paths'

function InformationRetrievalChat() {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState([
        {
            author: 'bot',
            content: (
                <Typography variant="body1">
                    Welcome to the Information Retrieval Chat! 👋
                    <br /><br />
                    Type in your query about patient data or disease information, and I will search our graph database for answers.
                    <br />
                    Example question: "How many patients have disease White-Kernohan syndrome?"
                    <br />
                    Please phrase each of your questions clearly to get the best results!
                </Typography>
            ),
            type: 'jsx'
        }
    ]);
    const [loading, setLoading] = useState(false);

    const handleInputChange = (event) => {
        setInput(event.target.value);
    };

    const sendMessage = async () => {
        if (!input.trim()) return;
        const userMessage = {
            author: 'clinician',
            content: input,
            type: 'text'
        };
        setMessages(messages => [...messages, userMessage]);
        setInput('');
        setLoading(true);

        try {
            const endpoint = `${ROOTS_Flask}/search-graph?question=${encodeURIComponent(input)}`;
            const response = await axios.post(endpoint);

            const botMessage = {
                author: 'bot',
                content: response.data.result,
                type: 'text'
            };
            setMessages(messages => [...messages, botMessage]);
        } catch (error) {
            setMessages(messages => [...messages, { author: 'bot', content: "I'm sorry, I can only provide assistance with generating a Cypher statement for querying a graph database based on the provided schema. Please try again.", type: 'text' }]);
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
            <Typography variant="h4" sx={{ textAlign: 'center' }}>Information Retrieval Chat</Typography>
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
                        boxSizing: 'border-box'
                    }}>
                        {message.type === 'jsx' ? message.content : <Typography variant="body1">{message.content}</Typography>}
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
                    InputLabelProps={{
                        style: { color: 'black' }
                      }}
                      InputProps={{
                        style: { color: 'black' },
                      }}
                />
                <Button variant="contained" onClick={sendMessage} disabled={!input.trim() || loading}>
                    Send
                </Button>
            </Box>
        </Box>
    );
}

export default InformationRetrievalChat;
