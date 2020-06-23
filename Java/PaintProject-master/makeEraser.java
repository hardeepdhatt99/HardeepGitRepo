package ca.utoronto.utm.paint;

import java.awt.event.MouseEvent;

public class makeEraser implements ShapeManipulatorStrategy {

	private Eraser Shape = new Eraser();
	private Eraser tempShape = new Eraser();
	private Drawer drawer;
	
	makeEraser(Drawer drawer){
		super();
		this.drawer = drawer;
	}
	
	@Override
	public void click(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void press(MouseEvent e) {
		this.Shape = new Eraser();
		this.tempShape = new Eraser();
		this.Shape.setColor(this.drawer.getBackground());
		this.tempShape.setThickness(this.drawer.getThickness());
		this.tempShape.setColor(this.drawer.getBackground());
		this.tempShape.setThickness(this.drawer.getThickness());
		this.drawer.getModel().addTemp(tempShape);
	}

	@Override
	public void release(MouseEvent e) {
		Point p = new Point(e.getX(), e.getY());
		this.Shape.AddPoint(p);
		this.Shape.setColor(this.drawer.getBackground());
		this.Shape.setThickness(this.drawer.getThickness());
		this.drawer.getModel().addCommand(Shape);
		this.Shape = null;
	}

	@Override
	public void drag(MouseEvent e) {
		if( this.Shape != null){
			this.Shape.AddPoint(new Point(e.getX(), e.getY()));
			this.tempShape.AddPoint(new Point(e.getX(), e.getY()));
			this.tempShape.setColor(this.drawer.getBackground());
			this.tempShape.setThickness(this.drawer.getThickness());
			this.drawer.getModel().addTemp(tempShape);
			}
	}

	@Override
	public void move(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void reset() {
		// TODO Auto-generated method stub
		
	}

}
