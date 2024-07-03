import cv2

def visualize_batch(data, labels, output) -> None:
    for i in range(data.shape[0]):
        img = data[i].permute(1,2,0).numpy()
        label = labels[i].permute(1,2,0).numpy()
        pred = output[i].permute(1,2,0).numpy()
        cv2.imshow("image", img)
        cv2.imshow("label", label)
        cv2.imshow("prediction", pred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()