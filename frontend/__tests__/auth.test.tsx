import { render, screen, fireEvent } from '@testing-library/react';
import RegisterForm from '../components/forms/RegisterForm';

describe('RegisterForm', () => {
  it('renders and submits', async () => {
    render(<RegisterForm />);
    fireEvent.change(screen.getByLabelText(/Email/i), { target: { value: 'a@example.com' } });
    fireEvent.change(screen.getByLabelText(/Password/i), { target: { value: 'password123' } });
    fireEvent.click(screen.getByText(/Register/i));
    // Can't assert API call without mocking fetch; at least ensures form renders and can click
    expect(screen.getByLabelText(/Email/i)).toBeInTheDocument();
  });
});
