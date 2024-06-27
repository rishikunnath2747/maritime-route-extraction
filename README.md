# MARITIME ROUTE EXTRACTION BASED ON DENSITY-BASED SPATIAL CLUSTERING OF TRAJECTORIES

## Abstract

Maritime route extraction through trajectory mining plays a pivotal role in advancing safety, efficiency, security, and environmental sustainability within maritime transporta- tion. This process provides invaluable insights for a diverse array of stakeholders, in- cluding shipping companies, port authorities, government agencies, and environmental organizations. However, this task is inherently challenging due to the inherent freedom in ship navigation and the prevalence of incorrect or noisy data. This project focuses on route extraction utilizing Automatic Identification System (AIS) data, a crucial compo- nent of maritime vessel tracking.

The initial phase of the project entails pre-processing trajectory data to enhance its quality, involving the removal of outliers and correction of erroneous data points. Given the dynamic and occasionally unpredictable nature of maritime activities, ensuring the accuracy and reliability of input data is vital for subsequent analyses.

Subsequently, the project addresses the challenge of improving computational effi- ciency in clustering and route extraction by employing proximity filtering. This tech- nique involves filtering out trajectories closest to the input point within a specific area of interest defined by drawing a circle connecting the start and end points. By imple- menting proximity filtering, the project aims to streamline the process of identifying and extracting maritime routes, thereby enhancing overall efficiency in maritime trans-
portation management.

Once trajectories within a designated area of interest are identified, a variety of clustering algorithms, including MD-DBSCAN, DBSCAN, K-Means, and Traclus, are employed. Among these algorithms, MD-DBSCAN emerges as the most effective, leveraging both spatial distance and Course Over Ground (COG) angle to achieve su- perior and more refined clustering. This integrated approach enables MD-DBSCAN to generate clusters that more accurately reflect the underlying patterns in maritime tra- jectories. This choice contributes to the project’s overarching goal of optimizing route extraction processes and reducing computational time.

## How to test

To test the project, follow the steps given below: 

1. Go the project location and run the following commands to start the server (You will need to have python and node installed). The dataset (ais.csv) is available here: [Link to dataset](https://drive.google.com/drive/u/0/folders/1pBwniIhAIUC8BX-vE83G4FjTa4gldG5T)

```sh
    pip install pandas numpy scipy scikit-learn
```

```sh
    npm i
```

```sh
    node server.js
```

The output will specify the port the project is running in. For example, here the server is running on port 3000.

```sh
    Server is running on http://localhost:3000
```

2. Go to the server.js, and replace the correct port number here

```sh
    const response = await fetch('http://localhost:3000/calculateMidpoint', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ lat1, lon1, lat2, lon2 })
    });
```

3. Once the server is running, open **index.html** in a browser. From here you can give the coordinates as input and view the route on the map.


## Acknowledgements

This project uses [Leaflet.js](https://leafletjs.com/), an open-source JavaScript library for interactive maps.

Leaflet.js is available under the [BSD 2-Clause License](https://github.com/Leaflet/Leaflet/blob/master/LICENSE).

