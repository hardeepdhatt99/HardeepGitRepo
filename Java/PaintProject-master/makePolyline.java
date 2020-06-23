package ca.utoronto.utm.paint;

import java.awt.event.MouseEvent;

public class makePolyline implements ShapeManipulatorStrategy{
	private PolyLine Shape = new PolyLine();
	private PolyLine tempShape = new PolyLine();
	private Drawer drawer;

	makePolyline(Drawer drawer) {
		super();
		this.drawer = drawer;
	}

	@Override
	public void move(MouseEvent e) {
		if(this.tempShape != null){
			if(!this.tempShape.getSkip()) this.tempShape.setSecondPoint(new Point(e.getX(), e.getY()));
			this.drawer.getModel().addTemp(tempShape);
		}
	}

	@Override
	public void drag(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void click(MouseEvent e) {
		Shape.AddPoint(new Point(e.getX(), e.getY()));
		tempShape.setFirstPoint(new Point(e.getX(), e.getY()));
		if(Shape.getSkip()) {
			tempShape.setFirstPoint(new Point(-1,-1));
			tempShape.setSecondPoint(new Point(-1,-1));
			
		}
		this.Shape.setColor(this.drawer.getColor());
		this.Shape.setThickness(this.drawer.getThickness());
		this.tempShape.setColor(this.drawer.getColor());
		this.tempShape.setThickness(this.drawer.getThickness());
		if(Shape.size() > 1) {
			this.drawer.getModel().addCommand(Shape);
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
