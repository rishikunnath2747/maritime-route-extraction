const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
const port = 3000;

// Middleware to parse JSON bodies
app.use(bodyParser.json());

// Enable CORS
app.use(cors());

// Handle POST requests to '/calculateMidpoint' endpoint
app.post('/calculateMidpoint', (req, res) => {
    try {
        console.log('Coordinates received, beginning processing')
        const startTime = new Date();
        // Extract the coordinates from the request body
        const { lat1, lon1, lat2, lon2 } = req.body;

        // Call the Python script with the coordinates
        const { spawnSync } = require('child_process');

        const pythonProcess = spawnSync('python3', ['script.py', lat1, lon1, lat2, lon2]);

        if (pythonProcess.error) {
            console.error('Failed to start Python process:', pythonProcess.error);
            res.status(500).send('Internal server error');
        } 
        else {
            let result = pythonProcess.stdout.toString();
            const errorOutput = pythonProcess.stderr.toString();
            const trimmedString = result.trim().slice(1, -1);
            const coordinates = trimmedString.split('], [').map(pair => {
                const [lat, lon] = pair.split(',').map(parseFloat);
                return [lat, lon];
            });
            const filteredCoordinates = coordinates.filter(coord => !coord.some(isNaN));
            const endTime = new Date(); // End time
            const executionTime = endTime - startTime; 
            console.log("Processing complete.", typeof filteredCoordinates,'being sent')
            console.log("Execution time:", executionTime, "milliseconds");
            res.send(filteredCoordinates);
        }


    
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
