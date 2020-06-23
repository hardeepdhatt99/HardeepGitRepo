package ca.utoronto.utm.paint;

import java.awt.event.MouseEvent;

public class makeRectangle implements ShapeManipulatorStrategy{

	private Rectangle Shape;;
	private Rectangle tempShape;
	private Drawer drawer;
	
	makeRectangle(Drawer drawer) {
		super();
		this.drawer = drawer;
	}
	
	@Override
	public void click(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void press(MouseEvent e) {
		Point cornerOne = new Point(e.getX(), e.getY());
		Point cornerTwo = new Point(e.getX(), e.getY());
		this.Shape = new Rectangle(cornerOne, cornerTwo, 0, 0);
		this.tempShape = new Rectangle(cornerOne, cornerTwo, 0, 0);
	}

	@Override
	public void release(MouseEvent e) {
		if(this.Shape != null) {
			this.Shape.setColor(this.drawer.getColor());
			this.Shape.setThickness(this.drawer.getThickness());
			this.Shape.setCornerTwo(new Point(e.getX(), e.getY()));
			this.drawer.getModel().addCommand(this.Shape);
			if(this.drawer.getFilled()) 
				this.Shape.setfill(true);
			this.Shape = null;
		}
	}

	@Override
	public void drag(MouseEvent e) {
		if (this.tempShape != null) {
			this.tempShape.setCornerTwo(new Point(e.getX(), e.getY()));
			this.tempShape.setColor(this.drawer.getColor());
			this.tempShape.setThickness(this.drawer.getThickness());
			if(this.drawer.getFilled())
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
		// TODO Auto-generated method stub
		
	}
}