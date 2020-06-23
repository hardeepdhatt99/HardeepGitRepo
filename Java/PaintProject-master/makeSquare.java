package ca.utoronto.utm.paint;

import java.awt.event.MouseEvent;

public class makeSquare implements ShapeManipulatorStrategy{

	private Square Shape;
	private Square tempShape;
	private Drawer drawer;
	
	makeSquare(Drawer drawer) {
		super();
		this.drawer = drawer;
	}

	@Override
	public void click(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void press(MouseEvent e) {
		Point P1 = new Point(e.getX(), e.getY());
		this.Shape = new Square(P1, P1, 0);
		this.tempShape = new Square(P1, P1, 0);
	}

	@Override
	public void release(MouseEvent e) {
		if(this.Shape != null) {
			this.Shape.setColor(this.drawer.getColor());
			this.Shape.setThickness(this.drawer.getThickness());
			this.Shape.setPTwo((new Point(e.getX(), e.getY())));
			this.drawer.getModel().addCommand(this.Shape);
			if(this.drawer.getFilled()) 
				this.Shape.setfill(true);
			this.Shape = null;
		}
	}

	@Override
	public void drag(MouseEvent e) {
		if (this.tempShape != null) {
			this.tempShape.setPTwo(new Point(e.getX(), e.getY()));
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
