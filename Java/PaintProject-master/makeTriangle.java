package ca.utoronto.utm.paint;

import java.awt.event.MouseEvent;

public class makeTriangle implements ShapeManipulatorStrategy{
	private Triangle Shape = new Triangle();
	private Triangle tempShape = new Triangle();
	private Drawer drawer;
	
	makeTriangle(Drawer drawer) {
		super();
		this.drawer = drawer;
	}

	@Override
	public void move(MouseEvent e) {
		this.drawer.getModel().resetTemp();
		if(this.Shape.getSize() == 2) {
			Triangle temp = new Triangle(this.Shape.getCornerOne(),this.Shape.getCornerTwo(),new Point(e.getX(), e.getY()),this.drawer.getFilled());
			tempShape.setColor(this.drawer.getColor());
			tempShape.setThickness(this.drawer.getThickness());
			this.drawer.getModel().addTemp(temp);
		}else if(this.Shape.getSize() == 1) {
			this.tempShape = new Triangle(this.Shape.getCornerOne(),new Point(e.getX(), e.getY()),new Point(e.getX(), e.getY()),false);
			tempShape.setColor(this.drawer.getColor());
			tempShape.setThickness(this.drawer.getThickness());
			this.drawer.getModel().addTemp(tempShape);
		}
	}

	@Override
	public void drag(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void click(MouseEvent e) {
		if(this.Shape.getSize()<3) {
			this.Shape.setColor(this.drawer.getColor());
			this.Shape.setThickness(this.drawer.getThickness());
			this.Shape.addPoint(new Point(e.getX(), e.getY()));
			if(this.drawer.getFilled()) this.Shape.setfill(true);
		}
		if(this.Shape.getSize() == 3) {
			this.drawer.getModel().addCommand(this.Shape);
			this.Shape = new Triangle();
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
