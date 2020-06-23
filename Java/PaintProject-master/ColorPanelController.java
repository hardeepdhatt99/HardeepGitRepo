package ca.utoronto.utm.paint;

import java.awt.Color;
import java.awt.Component;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;

public class ColorPanelController implements ActionListener{

	private DataColorPanel data;

	public ColorPanelController( DataColorPanel data ){
		this.data = data;
}

	
@Override
public void actionPerformed(ActionEvent e) {

	JButton b = (JButton) e.getSource();
	Color c = b.getBackground();
	data.setColor(c);
		
	}
}

