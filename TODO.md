GOAL:
Model that will, given a picture of the upper body of a person, determine how many fingers they are holding up
This requires two different CNNs.

1:
A CNN that takes the entire image, and returns a smaller image containing just the hand

Steps:

    Training:

        Training Data:
        - Need many different images, with sample boxes.
            - Tool that allows me to manually select a box around a hand
            - Many sample images
                - Tool that, given a video, returns all of the frames as images
                - Sample video of mine, amandas, and aarav's hands
        - Preprocess data
            - Gaussian blur, to avoid overfitting
            - ETC.

2:
A CNN that takes the hand image, and returns the number of digits held up


FULL INPUT->OUTPUT MAP:

Image of full body
Model takes full image, produces coordinates for box around hand
Function takes full image, box, produces smaller image
Model takes smaller image, produces number of digits help up

Methods needed for production input->output map:

