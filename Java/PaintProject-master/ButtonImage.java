package ca.utoronto.utm.paint;

import java.awt.GridLayout;

import javax.swing.ImageIcon;
import javax.swing.JButton;

public class ButtonImage implements Buttons {
	public ImageIcon[] Images() {
		ImageIcon rectangleImage = new ImageIcon(getClass().getResource("rectangle.png"));
		ImageIcon circleImage = new ImageIcon(getClass().getResource("circle.png"));
		ImageIcon squareImage = new ImageIcon(getClass().getResource("square.png"));
		ImageIcon triangleImage = new ImageIcon(getClass().getResource("triangle.png"));
		ImageIcon squiggleImage = new ImageIcon(getClass().getResource("draw.png"));
		ImageIcon polyImage = new ImageIcon(getClass().getResource("polyline.png"));
		ImageIcon fillImage = new ImageIcon(getClass().getResource("fill.png"));
		ImageIcon EraseImage = new ImageIcon(getClass().getResource("Eraser.png"));
		ImageIcon TextImage = new ImageIcon(getClass().getResource("Text.png"));
		
		ImageIcon[] icons = { circleImage, rectangleImage, squareImage, triangleImage, squiggleImage, EraseImage, polyImage, fillImage, TextImage};
		
		return icons;
		}
	public String[] labels() {
		String[] buttonLabels = { "Circle", "Rectangle", "Square", "Triangle", "Squiggle", "Erase", "Polyline", "Fill", "Text"};
		return buttonLabels;
	}
	
	public String[] tips() {
		String[] Tooltips = {"<html>Click and drag to create a Circle. Where you click is the center of the circle,<br>"
				+ "where you release will be on the circumference.</html>"
				,"<html>Click and drag to create a Rectangle, the first click will be a corner,<br> where you release will be the opposite corner.</html>"
				,"<html>Click and drag to create a Square, the first click will be on a corner,<br> where you release will be on a side.</html>"
				,"<html>Each click will make a vertex, once you have clicked three times the triangle will be made.</html>"
				,"<html>Anywhere you click will make some Squiggle.<br> Whenever you drag you will make Squiggles along your mouse.</html>"
				, "<html>Click and drag to erase shapes and lines on the paint panel."
				+ "<br>The eraser has an adjustable thickness.</html>"
				,"<html>First click will be the first point, each click after<br> will make another point with a Polyline inbetween.</html>"
				+ "If you double click the current Polyline will end, If you switch to another tool the current Polyline will end."
				,"<html>Fill works on Circle, Rectangle, Square, Triangle.<br> If selected it toggles Fill and new Shapes of the above type</html>"
				+ "a filled in Shape will be created."
				, "<html>Inserts Text by first entering the desired text in the top panel<br>"
				+ "Then click anywhere on the canvas to make the text appear<html>"};
		return Tooltips;
	}


}
