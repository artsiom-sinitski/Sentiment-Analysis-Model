
# Sentiment-Analysis-Model

Determine mood of a short text (tweet, review, etc.) using Convolutional Neural Network

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [Team](#team)
- [FAQ](#faq)
- [Support](#support)
- [License](#license)

## Installation

#### Install Python 3.7+
#### Install the following Python packages:
- Keras (DL high level framework)
- Tensorflow (DL backend)
- pandas (data manipulation)
- scikit-learn (model crossvalidation)
- matplotlib (data visualization)

## Clone

- Clone this repo to your local machine using `https://github.com/artsiom-sinitski/Sentiment-Analysis-Model.git`

## Features

## Usage

- First, prepare the data for the model. In the command line type: 
```
$> python prepData.py
```
- Second, train the model by running from the command line:
```
$> python trainModel.py default 7 32
```
       'default' is the name of your model  
       '120' is the number of epochs for training  
       '1' is number of batches  

- Third, make the model predict by running from the command line:
```
    $> python predictTextMood.py default 7 32
OR to make the model read tweets from "Tweets_samples.txt" file, type in:
    $> cat ./data/Tweets_samples.txt | python predictTextMood.py default 7 32
```
## Team

| <a href="https://github.com/artsiom-sinitski" target="_blank">**Artsiom Sinitski**</a> |
| :---: |
| [![Artsiom Sinitski](https://github.com/artsiom-sinitski)](https://github.com/artsiom-sinitski)|
| <a href="https://github.com/artsiom-sinitski" target="_blank">`github.com/artsiom-sinitski`</a> |

## FAQ

## Support

Reach out to me at the following places:
- <a href="https://github.com/artsiom-sinitski" rel="noopener noreferrer" target="_blank">GitHub account</a>
- <a href="https://www.instagram.com/artsiom_sinitski/" rel="noopener noreferrer" target="_blank"> Instagram account</a>

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright ©2019 
