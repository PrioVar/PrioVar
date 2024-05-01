import React, { useState, useEffect } from 'react';
import { Box, TextField, Button, CircularProgress, Typography, Paper } from '@mui/material';
import axios from 'axios';
import { ROOTS_Flask, ROOTS_PrioVar } from '../../routes/paths'

function InformationRetrievalChat() {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const welcomeMessage = {
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
        };
        setMessages([welcomeMessage]);
        retrieveChatHistory();
    }, []);

    const retrieveChatHistory = async () => {
        const medicalCenterId = localStorage.getItem('healthCenterId');
        if (!medicalCenterId) {
            console.log('No medical center ID found in local storage');
            return;
        }
        try {
            setLoading(true);
            const response = await axios.get(`${ROOTS_PrioVar}/chat/getGraphChatsByMedicalCenterId/${medicalCenterId}`);
            const chatHistoryMessages = response.data.flatMap(chat => [
                {
                    author: 'clinician',
                    content: chat.question,
                    type: 'text',
                    timestamp: formatTime(chat.timestamp)
                },
                {
                    author: 'bot',
                    content: chat.response,
                    type: 'text',
                    timestamp: formatTime(chat.timestamp)
                }
            ]);
            setMessages(messages => [...messages, ...chatHistoryMessages]);
        } catch (error) {
            console.error('Failed to retrieve chat history:', error);
            // You might want to handle this error more gracefully in a user-facing app
        } finally {
            setLoading(false);
        }
    };

    const handleInputChange = (event) => {
        setInput(event.target.value);
    };

    const formatTime = (timestamp) => {
        console.log(timestamp)
        const date = new Date(timestamp);
    
        const hours = date.getHours();
        const minutes = date.getMinutes();
    
        // Return formatted string, padding minutes with leading zero if necessary
        return `${hours}:${minutes.toString().padStart(2, '0')}`;
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
            // const endpoint = `${ROOTS_Flask}/search-graph?question=${encodeURIComponent(input)}`;
            //const response = await axios.post(endpoint);
            const response = await axios.post(`${ROOTS_Flask}/search-graph`, {
                question: input,
                healthCenterId: localStorage.getItem('healthCenterId')
            });

            const botMessage = {
                author: 'bot',
                content: response.data.result,
                type: 'text',
                timestamp: formatTime(response.data.timestamp)
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
                    <Box key={index} sx={{ margin: 1, textAlign: message.author === 'clinician' ? 'right' : 'left' }}>
                        <Paper sx={{
                            padding: 2,
                            bgcolor: message.author === 'clinician' ? '#e0f7fa' : '#fce4ec',
                            overflowWrap: 'break-word',
                            boxSizing: 'border-box',
                        }}>
                            {message.type === 'jsx' ? message.content : <Typography variant="body1">{message.content}</Typography>}
                        </Paper>
                        <Typography variant="caption" sx={{
                            display: 'block', // Ensure it's a block to appear on a new line
                            color: 'black',
                            fontSize: '0.75rem',
                            marginTop: '1px' // Add a little space between the paper and the timestamp
                        }}>
                            {message.timestamp}
                        </Typography>
                    </Box>
                ))}
                {loading && <CircularProgress sx={{ display: 'block', margin: 'auto' }} />}
            </Box>
            <Paper sx={{ display: 'flex', alignItems: 'center', gap: 1, p:1 }}>
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
            </Paper>
        </Box>
    );
}

export default InformationRetrievalChat;
