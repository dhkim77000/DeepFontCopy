# DeepFontCopy
### Copying font of logos, pictures or etc using neural network

##### The model is based on Multi-Content GAN for Few-Shot Font Style Transfer; Samaneh Azadi, Matthew Fisher, Vladimir Kim, Zhaowen Wang, Eli Shechtman, Trevor Darrell, in arXiv, 2017.

# Motivation
---
When editing photos or artworks using photoshop, designers often have to use the same or similar font used in brand logos or artworks. Even though the photoshop provides search tool that finds the most similar fonts in their library, it's not enough. Using GAN network by Azadi and some image processing techniques, We could copy the font of the given image. 

A page from the speakeasy magazine
![image](https://user-images.githubusercontent.com/89527573/174338402-6152dd9b-4974-4685-bec5-43d43c75fab2.png)



# How it works
---
![image](https://user-images.githubusercontent.com/89527573/174338509-4dde91e4-fe57-45dd-b7f6-7650068dd802.png)
The structure of the program is like this. The program segmentates the text from the given image and after some process, feed them as the input of GAN models.
![image](https://user-images.githubusercontent.com/89527573/174338555-87736fb1-3761-48ca-b519-a3bac84cc185.png)
# Results
---
## Le Labo
![image](https://user-images.githubusercontent.com/89527573/174338690-b591da65-43b3-40e0-95a8-3dcd31339e9f.png)
![image](https://user-images.githubusercontent.com/89527573/174338726-3609266b-6e11-48fd-86f9-656ef0795341.png)

## Artwork
![image](https://user-images.githubusercontent.com/89527573/174338787-907fbef3-38b7-4ab4-a602-119c735de383.png)
![image](https://user-images.githubusercontent.com/89527573/174338950-89be36a3-3bbf-47b1-8ead-2775017bfb45.png)

## Basquiat
![image](https://user-images.githubusercontent.com/89527573/174339034-702536dd-f900-42f3-9a82-b1d15ee6c550.png)
![image](https://user-images.githubusercontent.com/89527573/174339048-6c6359c4-2918-4a88-b012-e8b83fa4cd23.png)
