import React, { useState, useEffect } from 'react';
import { Box, TextField, Button, CircularProgress, Typography, Paper } from '@mui/material';
import axios from 'axios';
import { ROOTS_Flask, ROOTS_PrioVar } from '../../routes/paths'


function AISupportChat() {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const welcomeMessage = {
            author: 'bot',
            content: (
                <Typography variant="body1">
                    Welcome to the Clinician AI Support Chat! 👋
                    <br /><br />
                    Here's how I can assist you:
                    <br />
                    - You can ask me clinical questions, and I will interpret them using the PICO framework (Patient, Intervention, Comparison, Outcome).
                    <br />
                    - Then, I'll search the PubMed database to find relevant articles and present you with a summary.
                    <br />
                    - I utilize a vector database to understand the context of your question and provide the most accurate AI-generated suggestions based on RAG (Retrieval-Augmented Generation) method.
                    <br /><br />
                    Please note: I'm designed to respond to clinical questions like "How should the treatment be of a patient with tennis elbow?" So make sure to phrase your inquiries accordingly!
                    <br />
                    Just type your question below and press "Send" to begin our conversation 😊
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
            const response = await axios.get(`${ROOTS_PrioVar}/chat/getChatsByMedicalCenterId/${medicalCenterId}`);
            const chatHistoryMessages = response.data.flatMap(chat => [
                {
                    author: 'clinician',
                    content: chat.question,
                    type: 'text',
                    timestamp: formatTime(chat.timestamp)
                },
                {
                    author: 'bot',
                    content: (<Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                                PICO version of the question:
                              </Typography>),
                    contentDetail: (<span>{chat.pico_clinical_question}</span>),
                    type: 'jsx'
                },
                {
                    author: 'bot',
                    content: (<Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                                Number of relevant articles found in the PubMed database:
                              </Typography>),
                    contentDetail: (<span>{chat.article_count}</span>),
                    type: 'jsx'
                },
                {
                    author: 'bot',
                    content: (<Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                                Titles of the relevant articles:
                              </Typography>),
                    contentDetail: (<ul>{chat.article_titles.map((title, index) => (
                                        <li key={index}>{title}</li>
                                    ))}</ul>),
                    type: 'jsx'
                },
                {
                    author: 'bot',
                    content: (<Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                                AI suggestion:
                              </Typography>),
                    contentDetail: (<span>{chat.rag_GPT_output}</span>),
                    type: 'jsx',
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

    const formatTime = (timestamp) => {
        console.log(timestamp)
        const date = new Date(timestamp);
    
        const hours = date.getHours();
        const minutes = date.getMinutes();
    
        // Return formatted string, padding minutes with leading zero if necessary
        return `${hours}:${minutes.toString().padStart(2, '0')}`;
    };

    const handleInputChange = (event) => {
        setInput(event.target.value);
    };

    const sendMessage = async () => {
        if (!input.trim()) return;  // Prevent sending empty messages
        const clinicianMessage = {
            author: 'clinician',
            content: input,
            type: 'text',
            timestamp: formatTime(Date.now())
        };
        setMessages(messages => [...messages, clinicianMessage]); // Update the message list with the new message
        setInput(''); // Clear the input field
        setLoading(true); // Show loading indicator

        try {
            const response = await axios.post(`${ROOTS_Flask}/ai-help`, {
                clinical_question: clinicianMessage.content,
                healthCenterId: localStorage.getItem('healthCenterId')
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
                type: 'jsx',
                timestamp: formatTime(response.data.timestamp)
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
                    <Box key={index} sx={{ margin: 1, textAlign: message.author === 'clinician' ? 'right' : 'left' }}>
                        <Paper key={index} sx={{
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

export default AISupportChat;
