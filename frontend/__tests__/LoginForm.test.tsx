import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import LoginForm from '../components/forms/LoginForm';
import { signIn } from 'next-auth/react';

jest.mock('next-auth/react', () => ({
  signIn: jest.fn(),
}));

describe('LoginForm', () => {
  beforeEach(() => {
    (signIn as jest.Mock).mockReset();
  });

  it('calls signIn with credentials and redirects on success', async () => {
    (signIn as jest.Mock).mockResolvedValue({ ok: true });

    render(<LoginForm />);

    fireEvent.change(screen.getByLabelText(/Email/i), { target: { value: 'a@example.com' } });
    fireEvent.change(screen.getByLabelText(/Password/i), { target: { value: 'password123' } });

    fireEvent.click(screen.getByText(/Login/i));

    await waitFor(() => {
      expect(signIn).toHaveBeenCalledWith('credentials', expect.objectContaining({ email: 'a@example.com' , password: 'password123', redirect: false }));
    });
  });

  it('shows error when signIn returns an error', async () => {
    (signIn as jest.Mock).mockResolvedValue({ error: 'Invalid credentials' });

    render(<LoginForm />);

    fireEvent.change(screen.getByLabelText(/Email/i), { target: { value: 'a@example.com' } });
    fireEvent.change(screen.getByLabelText(/Password/i), { target: { value: 'badpass' } });

    fireEvent.click(screen.getByText(/Login/i));

    await waitFor(() => {
      expect(screen.getByText(/Invalid credentials/i)).toBeInTheDocument();
    });
  });
});
