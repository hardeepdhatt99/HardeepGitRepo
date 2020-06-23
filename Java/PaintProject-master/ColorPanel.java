package ca.utoronto.utm.paint;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.util.Observable;
import java.util.Observer;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

public class ColorPanel extends JPanel implements Observer{

	JButton ColorNow = new JButton();
	JButton OwnColor = new JButton();
	private int[] panelColors = new int[3];
	JTextField f1 = new JTextField("");
	JTextField f2 = new JTextField("");
	JTextField f3 = new JTextField("");
	
	public ColorPanel(DataColorPanel data){
	
	this.setLayout(null);
	JButton blue = new JButton();
	blue.setBackground(Color.blue);
	blue.addActionListener(new ColorPanelController(data));
	blue.setBounds(5, 5, 22, 22);
	JButton red = new JButton();
	red.setBackground(Color.red);
	red.addActionListener(new ColorPanelController(data));
	red.setBounds(30, 5, 22, 22);
	JButton black = new JButton();
	black.setBackground(Color.black);
	black.addActionListener(new ColorPanelController(data));
	black.setBounds(55, 5, 22, 22);
	JButton white = new JButton();
	white.setBackground(Color.white);
	white.addActionListener(new ColorPanelController(data));
	white.setBounds(80, 5, 22, 22);
	JButton orange = new JButton();
	orange.setBackground(Color.orange);
	orange.addActionListener(new ColorPanelController(data));
	orange.setBounds(5, 30, 22, 22);
	JButton yellow = new JButton();
	yellow.setBounds(30, 30, 22, 22);
	yellow.setBackground(Color.yellow);
	yellow.addActionListener(new ColorPanelController(data));
	JButton green = new JButton();
	green.setBounds(55, 30, 22, 22);
	green.setBackground(Color.green);
	green.addActionListener(new ColorPanelController(data));
	JButton gray = new JButton();
	gray.setBounds(80, 30, 22, 22);
	gray.setBackground(Color.gray);
	gray.addActionListener(new ColorPanelController(data));
	
	this.setLayout(new GridLayout(1,3));
	
	//JPanel Special
	JPanel Special = new JPanel();
	
	ColorNow.setBackground(Color.BLACK);
	OwnColor.setBackground(Color.WHITE);
	JLabel L1 = new JLabel("Current");
	L1.setBounds(5, 35, 50, 10);
	JLabel L2 = new JLabel("Previous");
	L2.setBounds(55, 35, 50, 10);
	Special.add(L1);
	Special.add(L2);
	ColorNow.setBounds(13, 0, 30, 30);
	OwnColor.setBounds(65, 0, 30, 30);
	Special.add(ColorNow);
	Special.add(OwnColor);
	Special.setPreferredSize(new Dimension(30, 35));
	Special.setLayout(null);
	
	this.add(Special);
	
	//JPanel ColorSelections
	JPanel ColorSelections = new JPanel();
	ColorSelections.setLayout(null);
	ColorSelections.add(blue);
	ColorSelections.add(red);
	ColorSelections.add(black);
	ColorSelections.add(white);
	ColorSelections.add(green);
	ColorSelections.add(yellow);
	ColorSelections.add(orange);
	ColorSelections.add(gray);
	Special.setPreferredSize(new Dimension(100, 50));
	this.add(ColorSelections);
	
	//JPanel OwnerColor
	
	f1 = new JTextField("Red");
	f2 = new JTextField("Green");
	f3 = new JTextField("Blue");
	
	colorFocusListener l1 = new colorFocusListener();
	colorFocusListener l2 = new colorFocusListener();
	colorFocusListener l3 = new colorFocusListener();
	
	f1.addFocusListener(l1);
	f2.addFocusListener(l2);
	f3.addFocusListener(l3);

	l1.setColor("Red");
	l2.setColor("Green");
	l3.setColor("Blue");

	l1.addTextField(f1);
	l2.addTextField(f2);
	l3.addTextField(f3);
	
	JPanel OwnerColorPanel = new JPanel();
	OwnerColorPanel.setLayout(new GridLayout(2,2));
	OwnerColorPanel.add(f1);
	OwnerColorPanel.add(f2);
	OwnerColorPanel.add(f3);
	JButton b = new JButton("Set");
	
	b.addActionListener(new ActionListener(){
		int red;
		int green;
		int blue;
		
		public int[] getColors() {
			int[] colors = new int[3];
			colors[0] = this.red;
			colors[1] = this.green;
			colors[2] = this.blue;			
			return colors;
		}
		
    	public void actionPerformed(ActionEvent e){
    		try {
    			red = Integer.parseInt(f1.getText());
				if (red > 250){
					red = 250;
				}
				if (red < 0){
					red = 0;
				}
    		}
    		catch (NumberFormatException e1){
    			f1.setText("Red");
   				}
    		try {
    			green = Integer.parseInt(f2.getText());
				if (green > 250){
					green = 250;
				}
				if (green < 0){
					green = 0;
				}
    		}
    		catch (NumberFormatException e1){
    			f2.setText("Green");
   				}
    		try {
    			blue = Integer.parseInt(f3.getText());
				if (blue > 250){
					blue = 250;
				}
				if (blue < 0){
					blue = 0;
				}
    		}
    		catch (NumberFormatException e1){
    			f3.setText("Blue");
   				}
    		
    		try {
    			Color c = new Color(red, green, blue);
    			data.setColor(c);
    		}
    		catch(Exception error) {
    			
    		}
	}
	});
	OwnColor.addActionListener(new ColorPanelController(data));
	OwnerColorPanel.add(b);
	this.add(OwnerColorPanel);
	}

	@Override
	public void update(Observable arg0, Object arg1) {
		OwnColor.setBackground(ColorNow.getBackground());
		ColorNow.setBackground((Color)arg1);
	}
}
