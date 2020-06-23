package ca.utoronto.utm.paint;

import java.awt.event.MouseEvent;

public class makeText implements ShapeManipulatorStrategy{

	private Text Shape;
	private Text tempShape;
	private Drawer drawer;
	
	makeText(Drawer drawer) {
		super();
		this.drawer = drawer;
	}

	@Override
	public void move(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void drag(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void click(MouseEvent e) {
		Point p = new Point(e.getX(), e.getY());
		this.Shape = new Text(this.drawer.getText(), p);
		this.Shape.setSize(this.drawer.getThickness());
		this.Shape.setColor(this.drawer.getColor());
		if (this.drawer.getText() != "") {
		this.drawer.getModel().addCommand(Shape);
		this.Shape = null;
		}
	}

	@Override
	public void press(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void release(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void reset() {
		// TODO Auto-generated method stub
		
	}

}
