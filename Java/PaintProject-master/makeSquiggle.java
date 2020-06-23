package ca.utoronto.utm.paint;

import java.awt.event.MouseEvent;

public class makeSquiggle implements ShapeManipulatorStrategy{

	private Squiggle Shape;
	private Squiggle tempShape;
	private Drawer drawer;
	
	makeSquiggle(Drawer drawer) {
		super();
		this.drawer = drawer;
	}

	@Override
	public void move(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void drag(MouseEvent e) {
		if( this.Shape != null){
			this.Shape.AddPoint(new Point(e.getX(), e.getY()));
			this.tempShape.AddPoint(new Point(e.getX(), e.getY()));
			this.tempShape.setColor(this.drawer.getColor());
			this.tempShape.setThickness(this.drawer.getThickness());
			this.drawer.getModel().addTemp(tempShape);
		}
		
	}

	@Override
	public void click(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void press(MouseEvent e) {
		this.Shape= new Squiggle(); 
		this.tempShape = new Squiggle();
		this.drawer.getModel().addTemp(tempShape);
	}

	@Override
	public void release(MouseEvent e) {
		Point p = new Point(e.getX(), e.getY());
		this.Shape.AddPoint(p);
		this.Shape.setColor(this.drawer.getColor());
		this.Shape.setThickness(this.drawer.getThickness());
		this.drawer.getModel().addCommand(Shape);
		this.Shape = null;
	}

	@Override
	public void reset() {
		// TODO Auto-generated method stub
		
	}

}
