# Graph Visualization App

This is a simple JavaFX app where you can load images and build graphs from them. You can do things like find paths, compare graphs, and even classify them.

---

## What it does

- Load an image from your computer.
- Make a graph from the image.
- Show the graph on the screen.
- Find the shortest path between two points you pick.
- Build Minimum Spanning Tree (MST) and k-Nearest Neighbor (KNN) graphs.
- Compare two graphs side by side.
- Train and classify graphs with labels.
- Show progress and status messages.
- Shows errors if something goes wrong.

---

## How to use

1. Load an image by clicking the button.
2. Build a graph from the image.
3. Use buttons to do things like MST, KNN, find paths, compare graphs, or classify.
4. For finding path, you click start and end points on the image popup.
5. See results and messages in the text area.
6. If there’s an error, a popup will tell you what went wrong.

---

## What it looks like

- Tabs to switch between graph view and comparison.
- Scroll panes to see graphs (you can zoom and pan).
- Buttons to run different functions.
- Status bar at the bottom showing what’s happening.
- Text area logs all the messages.

---

## How it works (sort of)

- It uses JavaFX for the UI and drawing graphs.
- The graphs are made from nodes and edges.
- You can zoom and pan the graph view.
- When you compare graphs, they get scaled to fit side by side.
- The app logic handles graph calculations and background work.
- UI updates are done carefully on the JavaFX thread.

---

## How to run

- Make sure you have Java and JavaFX installed.
- Compile and run the app.
- Use the buttons in the UI to do stuff.

---

## Stuff to improve later

- Make K in KNN changeable by user.
- Save and load graphs.
- Speed up heavy calculations with threads.
- Better UI and more graph algorithms.
- Support more image types.

---

## Notes

This is a beginner-level project. If you find bugs or want to add features, feel free to reach out!

---

Glad you stopped by to see this!
