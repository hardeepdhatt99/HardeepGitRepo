package ca.utoronto.utm.paint;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Observable;
import java.util.Observer;
import java.util.Stack;

import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

public class HistoryPanel  extends JPanel implements ActionListener, Observer{
	private View view; // So we can talk to our parent or other components of the view
	private ArrayList<String> History = new ArrayList<String>();
	
	public HistoryPanel(View view) {	
		this.view=view;
		resetPanel();
	}
	
	private void undoTextField(JTextField T) {
		T.setEnabled(false);
		T.setHorizontalAlignment(SwingConstants.CENTER);
		T.setBackground(new Color(100,100,100));
		T.setForeground(new Color(0,0,0));
	}
	
	private void formatTextField(JTextField T) {
		T.setEditable(false);
		T.setHorizontalAlignment(SwingConstants.CENTER);
		T.setBackground(new Color(255,255,255));
		T.setForeground(new Color(0,0,0));
	}
	
	private void currentOperation(JTextField T) {

		T.setEditable(false);
		T.setHorizontalAlignment(SwingConstants.CENTER);
		T.setBackground(new Color(153, 204, 255));
		T.setForeground(new Color(0,0,0));
	}
	
	public void resetPanel(){
		this.removeAll();
		this.setPreferredSize(new Dimension(93,592));
		this.setLayout(new GridLayout(10,0));
		JButton H = new JButton("History");
		H.setEnabled(false);
		H.setOpaque(true);
		H.setHorizontalAlignment(SwingConstants.CENTER);
		H.setBackground(new Color(255,255,255));
		H.setForeground(new Color(0,0,0));
		this.add("", H);
		int numUndos = 1;
		for(String undo:History) if(undo.contains("Undo")) numUndos++;
		for(int i = 0; i<this.History.size(); i++) {
			JTextField T = new JTextField(this.History.get(i));
			T.setText(this.History.get(i));
			if(T.getText().contains("Undo")) {
				undoTextField(T);
			}
			else formatTextField(T);
			if(i == this.History.size()-numUndos)currentOperation(T);
			this.add(T);
		}
		this.view.revalidate();
		this.view.repaint();
	}

	public void setHistory(ArrayList<String> H) {
		this.History = H;
		resetPanel();
	}
	
	@Override
	public void actionPerformed(ActionEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void update(Observable arg0, Object arg1) {
		Stack<String> Order = ((PaintModel) arg0).getOrder();
		ArrayList<String> History = new ArrayList<String>();
		for(String s:Order) History.add(s);
		while(History.size() > 9) {
			History.remove(0);
		}
		if(Order.size() != 0)
			this.setHistory(History);
	}

	
}
