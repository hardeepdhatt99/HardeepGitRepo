package ca.utoronto.utm.paint;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

// https://docs.oracle.com/javase/8/docs/api/java/awt/Graphics2D.html
// https://docs.oracle.com/javase/tutorial/2d/

class ShapeChooserPanel extends JPanel implements ActionListener {
	private View view; // So we can talk to our parent or other components of the view
	private JLabel status;
	private ImageIcon[] icons;
	private String[] buttonLabels;
	private String[] Tooltips;
	public ShapeChooserPanel(View view) {	
		this.view=view;
	
		ButtonFactory button1 = new ButtonFactory();
		Buttons image =	button1.getButton("ButtonImage");
		icons = image.Images();
		buttonLabels = image.labels();
		Tooltips = image.tips();
		
		this.setLayout(new GridLayout(buttonLabels.length + 1, 1));
		int i = 0;
		for (String label : buttonLabels) {
			JButton button = new JButton(label);
			button.setIcon(icons[i]);
			this.add(button);
			button.addActionListener(this);
			button.setToolTipText(Tooltips[i]);
			i++;
		}
		this.status = new JLabel("Drawing Mode: ");
		this.add(this.status);
		
	}
	
	/**
	 * Controller aspect of this
	 */
	@Override
	public void actionPerformed(ActionEvent e) {
		if(e.getActionCommand() != "Fill") {
			this.view.getPaintPanel().setMode(e.getActionCommand());
				if (this.view.getPaintPanel().getFill() && (e.getActionCommand() == "Rectangle" || e.getActionCommand() == "Square" || e.getActionCommand() == "Circle"))
					{
					this.status.setText("Drawing Mode: " + e.getActionCommand()+ " Fill");
					}
				else
					this.status.setText("Drawing Mode: " + e.getActionCommand());
					
		}
		else {
			this.view.getPaintPanel().toggleFill();
			if (!this.view.getPaintPanel().getFill()){
				String s = this.status.getText();
				if (s.contains("Fill"))
					s = s.substring(0, s.length()-5);
				this.status.setText(s);
			}else {
				String s = this.status.getText();
				if(!s.contains("Polyline") && !s.contains("Squiggle") && !s.contains("Text") && !s.contains("Erase"))
				this.status.setText(this.status.getText()+" Fill");
			}
		}
	}

	
}
