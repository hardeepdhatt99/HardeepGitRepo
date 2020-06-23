package ca.utoronto.utm.paint;

import java.awt.event.MouseEvent;

public class makeCircle implements ShapeManipulatorStrategy{
	private Circle Shape;
	private Circle tempShape;
	private Drawer drawer;
	
	makeCircle(Drawer drawer) {
		super();
		this.drawer = drawer;
	}


	@Override
	public void click(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void press(MouseEvent e) {
		Point centre = new Point(e.getX(), e.getY());
		this.Shape=new Circle(centre, 0);
		this.tempShape = new Circle(centre, 0);
	}


	@Override
	public void release(MouseEvent e) {
		if(this.Shape!=null){
			this.Shape.setColor(this.drawer.getColor());
			this.Shape.setThickness(this.drawer.getThickness());
			this.Shape.setRadiusP(e.getX(), e.getY());
			this.drawer.getModel().addCommand(this.Shape);
			if(this.drawer.getFilled()) 
				this.Shape.setfill(true);
			this.Shape=null;
		}
	}

	@Override
	public void drag(MouseEvent e) {
		
		if (this.tempShape != null) {
			this.tempShape.setRadiusP(e.getX(), e.getY());
			this.tempShape.setColor(drawer.getColor());
			this.tempShape.setThickness(drawer.getThickness());
			if(drawer.getFilled())
				tempShape.setfill(true);
			this.drawer.getModel().addTemp(this.tempShape);
			}
	}

	@Override
	public void move(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void reset() {
	}

}
