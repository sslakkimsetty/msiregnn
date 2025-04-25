# import matplotlib
# matplotlib.use('Qt5Agg')  # or 'Qt5Agg'

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
# from matplotlib.widgets import PolygonSelector

class InteractiveROI:
    def __init__(self, img):
        self.img = img
        self.roi_mask = np.zeros_like(img, dtype=bool)
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(img, cmap='gray')
        self.poly_selector = PolygonSelector(self.ax, self.onselect, useblit=True)
        # self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        plt.show()

    def onselect(self, verts):
        path = Polygon(verts)
        self.roi_mask[:] = 0
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                if path.contains_point((j, i)):
                    self.roi_mask[i, j] = True

    def onkey(self, event):
        if event.key == 'enter':
            plt.close(self.fig)

    def get_roi_mask(self):
        return self.roi_mask

def draw_roi_on_image(img):
    """
    Select a single region-of-interest (ROI) on an optical image.

    Parameters:
    img (numpy.ndarray): Image array to select ROI on.

    Returns:
    numpy.ndarray: ROI as a logical matrix of the same spatial dimensions as img.
    """
    roi_selector = InteractiveROI(img)
    return roi_selector.get_roi_mask()

def main():
    # Load or create a sample image
    img = np.random.random((100, 100))
    roi_mask = draw_roi_on_image(img)

    # Display the ROI mask
    plt.imshow(roi_mask, cmap='gray')
    plt.title('Selected ROI')
    plt.show()

if __name__ == "__main__":
    main()