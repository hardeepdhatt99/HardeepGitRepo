package ca.utoronto.utm.paint;

import java.awt.event.MouseEvent;

public interface ShapeManipulatorStrategy {
	shapes Shape = new shapes();
	shapes tempShape = new shapes();
	
	void move(MouseEvent e);
	void drag(MouseEvent e);
	void click(MouseEvent e);
	void press(MouseEvent e);
	void release(MouseEvent e);
	void reset();
	
}
